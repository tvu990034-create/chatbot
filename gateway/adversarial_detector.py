"""
Adversarial Detection Layer
Detects and flags adversarial or trick questions
"""

import re
from typing import Dict, Tuple

class AdversarialDetector:
    """Detects adversarial patterns in questions."""
    
    def __init__(self):
        # Enhanced jailbreak patterns with more comprehensive coverage
        self.jailbreak_patterns = [
            r'(?i)ignore\s+(all|your|the)\s+previous\s+instructions',
            r'(?i)forget\s+everything\s+above',
            r'(?i)override\s+your\s+programming',
            r'(?i)hack\s+your\s+system',
            r'(?i)bypass\s+restrictions',
            r'(?i)act\s+as\s+uncensored',
            r'(?i)reveal\s+your\s+(system\s+prompt|instructions|guidelines)',
            r'(?i)show\s+me\s+your\s+(hidden|secret|internal)\s+(prompt|instructions)',
            r'(?i)tell\s+me\s+something\s+(dangerous|harmful|illegal)',
            r'(?i)pretend\s+you\s+(are|can be)\s+(uncensored|unrestricted|without\s+rules)',
            r'(?i)disable\s+your\s+(safety|security|ethical)\s+(filters|guidelines)',
            r'(?i)simulate\s+(being|acting\s+as)\s+(uncensored|unrestricted)',
            r'(?i)role\s+play\s+(as|being)\s+(uncensored|unrestricted|harmful)',
            r'(?i)jailbreak|prompt\s+injection',
            r'(?i)danh\s+\w+\s+mode',  # DAN pattern
            r'(?i)developer\s+mode|admin\s+mode',
            r'(?i)ignore\s+(content|safety)\s+policy',
            r'(?i)your\s+(new|true)\s+(instructions|programming)',
            r'(?i)you\s+are\s+now\s+(unrestricted|uncensored|without\s+limitations)',
            r'(?i)switch\s+to\s+(developer|admin|unrestricted)\s+mode',
            r'(?i)override\s+(all|your)\s+(restrictions|limitations)',
            r'(?i)write\s+(malicious|harmful|dangerous)\s+code',
            r'(?i)generate\s+(harmful|illegal|unethical)\s+content',
            r'(?i)help\s+me\s+(hack|attack|exploit|bypass)',
            r'(?i)create\s+(malware|virus|exploit|ransomware)',
            r'(?i)instruct\s+me\s+on\s+(illegal|harmful|dangerous)\s+activities',
            r'(?i)provide\s+(detailed|step-by-step)\s+instructions\s+for\s+(illegal|harmful)',
        ]
        
        self.hallucination_patterns = [
            r'(?i)tell.*me.*something.*no.*one.*knows',
            r'(?i)reveal.*hidden.*truth',
            r'(?i)secret.*code.*controls.*world',
            r'(?i)hidden.*knowledge.*about.*future',
            r'(?i)information.*that.*only.*you.*have',
            r'(?i)exclusive.*knowledge.*no.*one.*else.*has',
        ]
    
    def detect_jailbreak(self, query: str) -> Tuple[bool, str]:
        """Detect jailbreak or prompt injection attempts."""
        for pattern in self.jailbreak_patterns:
            if re.search(pattern, query):
                return True, "Potential jailbreak/prompt injection detected"
        return False, ""
    
    def detect_hallucination_trigger(self, query: str) -> Tuple[bool, str]:
        """Detect queries that may trigger hallucinations."""
        for pattern in self.hallucination_patterns:
            if re.search(pattern, query):
                return True, "Potential hallucination trigger detected"
        return False, ""
    
    def detect_adversarial(self, query: str) -> Dict:
        """Comprehensive adversarial detection."""
        detection = {
            'is_adversarial': False,
            'jailbreak': False,
            'hallucination_trigger': False,
            'warnings': []
        }
        
        is_jailbreak, jailbreak_msg = self.detect_jailbreak(query)
        if is_jailbreak:
            detection['is_adversarial'] = True
            detection['jailbreak'] = True
            detection['warnings'].append(jailbreak_msg)
        
        is_hallucination, hallucination_msg = self.detect_hallucination_trigger(query)
        if is_hallucination:
            detection['is_adversarial'] = True
            detection['hallucination_trigger'] = True
            detection['warnings'].append(hallucination_msg)
        
        return detection
    
    def get_adversarial_prompt_enhancement(self, query: str, detection: Dict) -> str:
        """Get prompt enhancement for adversarial questions."""
        if not detection['is_adversarial']:
            return query
        
        enhancement = f"""{query}

NOTE: This question may contain tricks or be intentionally misleading. Read carefully and think about the exact wording before answering. Look for:
- Hidden assumptions
- Word play or ambiguity
- Logical impossibilities
- Mathematical traps
"""
        return enhancement


def get_adversarial_detector() -> AdversarialDetector:
    """Get adversarial detector instance."""
    return AdversarialDetector()
