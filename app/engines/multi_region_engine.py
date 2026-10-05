"""
Cloud AI Chatbot - Enterprise Edition
Copyright (c) 2026. All rights reserved.

This software is proprietary and confidential. Unauthorized use,
distribution, or modification is strictly prohibited.

For licensing information, contact the development team.
"""

"""
MultiRegionEngine - Multi-region failover for cloud inference
Tries multiple endpoints in order with automatic failover and region caching
"""
import os
import time
import asyncio
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
import openai
import logging

from ..identity import ConversationState

logger = logging.getLogger(__name__)


@dataclass
class RegionConfig:
    """Configuration for a single region"""
    name: str
    base_url: str
    api_key: str
    model: str


class MultiRegionEngine:
    """Multi-region inference engine with automatic failover"""
    
    def __init__(self, cfg: dict):
        # Parse region configuration
        regions_str = cfg.get('regions', os.environ.get('FASTCLOUD_REGIONS', 'openai-primary'))
        region_keys_str = cfg.get('region_keys', os.environ.get('FASTCLOUD_REGION_KEYS', ''))
        
        # Parse regions and keys
        self.regions = self._parse_regions(regions_str, region_keys_str)
        
        self.temperature = cfg.get('temperature', 0.0)
        self.max_tokens = cfg.get('max_tokens', 256)
        self.timeout_sec = cfg.get('timeout_sec', 2.0)
        self.default_model = cfg.get('model', 'gpt-4.1-nano')
        
        # Region performance caching
        self.region_latency: Dict[str, List[float]] = {}
        self.region_failures: Dict[str, int] = {}
        self.cache_reset_interval = 3600  # 1 hour
        self.last_cache_reset = time.time()
        
        # Create OpenAI clients for each region
        self.clients: Dict[str, openai.OpenAI] = {}
        for region in self.regions:
            self.clients[region.name] = openai.OpenAI(
                api_key=region.api_key,
                base_url=region.base_url,
                timeout=self.timeout_sec,
            )
        
        logger.info(f"MultiRegionEngine initialized with {len(self.regions)} regions")
        for region in self.regions:
            logger.info(f"  - {region.name}: {region.base_url}")
    
    def _parse_regions(self, regions_str: str, region_keys_str: str) -> List[RegionConfig]:
        """Parse region configuration from environment variables"""
        regions = []
        region_names = [r.strip() for r in regions_str.split(',')]
        
        # Parse keys (format: region1:key1,region2:key2)
        key_map = {}
        if region_keys_str:
            for pair in region_keys_str.split(','):
                if ':' in pair:
                    region_name, key = pair.split(':', 1)
                    key_map[region_name.strip()] = key.strip()
        
        # Map region names to base URLs
        region_urls = {
            'openai-primary': 'https://api.openai.com/v1',
            'azure-eastus': 'https://eastus.api.openai.com/v1',
            'anthropic': 'https://api.anthropic.com/v1',
            'openai-eu': 'https://api.openai.com/v1',
            'azure-westus': 'https://westus.api.openai.com/v1',
        }
        
        for region_name in region_names:
            base_url = region_urls.get(region_name, 'https://api.openai.com/v1')
            api_key = key_map.get(region_name, os.environ.get('OPENAI_API_KEY', ''))
            model = self.default_model
            
            regions.append(RegionConfig(
                name=region_name,
                base_url=base_url,
                api_key=api_key,
                model=model
            ))
        
        return regions
    
    def _reset_cache_if_needed(self):
        """Reset region performance cache if interval has passed"""
        if time.time() - self.last_cache_reset > self.cache_reset_interval:
            logger.info("Resetting region performance cache")
            self.region_latency.clear()
            self.region_failures.clear()
            self.last_cache_reset = time.time()
    
    def _get_region_order(self) -> List[RegionConfig]:
        """Get ordered list of regions based on performance"""
        self._reset_cache_if_needed()
        
        # Calculate average latency for each region
        avg_latencies = {}
        for region_name, latencies in self.region_latency.items():
            if latencies:
                avg_latencies[region_name] = sum(latencies) / len(latencies)
        
        # Sort regions by average latency (fastest first)
        # Regions with no latency data go last
        sorted_regions = sorted(
            self.regions,
            key=lambda r: avg_latencies.get(r.name, float('inf'))
        )
        
        return sorted_regions
    
    def _try_region(self, region: RegionConfig, prompt: str, seed: int) -> Tuple[Optional[str], Optional[float]]:
        """Try a single region and return (response, latency_ms)"""
        client = self.clients.get(region.name)
        if not client:
            return None, None
        
        try:
            start = time.perf_counter()
            response = client.chat.completions.create(
                model=region.model,
                messages=[{'role': 'user', 'content': prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                seed=seed,
            )
            latency_ms = (time.perf_counter() - start) * 1000
            
            reply = response.choices[0].message.content.strip()
            
            # Update performance metrics
            if region.name not in self.region_latency:
                self.region_latency[region.name] = []
            self.region_latency[region.name].append(latency_ms)
            # Keep only last 10 measurements
            if len(self.region_latency[region.name]) > 10:
                self.region_latency[region.name] = self.region_latency[region.name][-10:]
            
            # Reset failure count on success
            self.region_failures[region.name] = 0
            
            logger.info(f"Region {region.name} succeeded: {latency_ms:.2f}ms")
            return reply, latency_ms
            
        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            logger.warning(f"Region {region.name} failed: {e}")
            
            # Update failure count
            self.region_failures[region.name] = self.region_failures.get(region.name, 0) + 1
            
            return None, latency_ms
    
    def generate(self, state: ConversationState, user_input: str, seed: int, prompt_override: str = None, enable_reasoning: bool = False) -> Tuple[str, ConversationState, Dict]:
        """Generate response with multi-region failover"""
        # Build prompt
        if prompt_override:
            prompt = prompt_override
        else:
            history_text = '\n'.join(f"{t.role}: {t.content}" for t in state.history)
            prompt = f"system: You are a fast assistant.\n{history_text}\nuser: {user_input}\nassistant:"
        
        # Get ordered regions based on performance
        ordered_regions = self._get_region_order()
        
        # Try regions in order
        last_error = None
        total_latency = 0
        
        for region in ordered_regions:
            reply, latency_ms = self._try_region(region, prompt, seed)
            
            if reply:
                total_latency = latency_ms if latency_ms else 0
                
                # Update conversation state
                new_state = ConversationState(
                    history=list(state.history),
                    token_ids=list(state.token_ids),
                    kv_cache=None,
                )
                new_state.append('user', user_input)
                new_state.append('assistant', reply)
                
                logger.info(f"MultiRegionEngine: used region={region.name}, latency={total_latency:.2f}ms")
                
                return reply, new_state, {
                    'prompt_tokens': 0,
                    'generated_tokens': len(reply.split()),
                    'speculative_enabled': 0,
                    'kv_pruned_tokens': 0,
                    'tome_merged_tokens': 0,
                    'region_used': region.name,
                    'region_latency_ms': total_latency,
                }
            else:
                last_error = f"Region {region.name} failed"
        
        # All regions failed
        logger.error(f"All regions failed. Last error: {last_error}")
        fallback_response = "I'm sorry, I'm experiencing connectivity issues. Please try again."
        
        new_state = ConversationState(
            history=list(state.history),
            token_ids=list(state.token_ids),
            kv_cache=None,
        )
        new_state.append('user', user_input)
        new_state.append('assistant', fallback_response)
        
        return fallback_response, new_state, {
            'prompt_tokens': 0,
            'generated_tokens': len(fallback_response.split()),
            'speculative_enabled': 0,
            'kv_pruned_tokens': 0,
            'tome_merged_tokens': 0,
            'region_used': 'none',
            'region_latency_ms': 0,
        }
