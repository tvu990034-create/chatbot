"""
Data Ingestion System for Knowledge Base
Loads documents from various sources (text files, JSON, etc.) for the RAG system.
"""

import os
import json
import glob
from typing import List, Dict, Tuple, Optional
from pathlib import Path


class DataIngestion:
    """
    Handles loading and processing documents from various sources
    for the retrieval-augmented generation system.
    """
    
    def __init__(self, knowledge_base_path: str = "data/knowledge_base"):
        self.knowledge_base_path = knowledge_base_path
        self.supported_formats = {'.txt', '.json', '.md', '.csv'}
    
    def load_all_documents(self) -> Tuple[List[str], List[str]]:
        """
        Load all documents from the knowledge base directory.
        
        Returns:
            Tuple of (documents list, document IDs list)
        """
        documents = []
        doc_ids = []
        
        if not os.path.exists(self.knowledge_base_path):
            print(f"Warning: Knowledge base path {self.knowledge_base_path} does not exist")
            return documents, doc_ids
        
        # Load all supported files
        for file_path in glob.glob(os.path.join(self.knowledge_base_path, "**/*"), recursive=True):
            if os.path.isfile(file_path):
                ext = os.path.splitext(file_path)[1].lower()
                if ext in self.supported_formats:
                    docs, ids = self.load_file(file_path)
                    documents.extend(docs)
                    doc_ids.extend(ids)
        
        print(f"Loaded {len(documents)} documents from knowledge base")
        return documents, doc_ids
    
    def load_file(self, file_path: str) -> Tuple[List[str], List[str]]:
        """
        Load documents from a single file.
        
        Args:
            file_path: Path to the file to load
            
        Returns:
            Tuple of (documents list, document IDs list)
        """
        ext = os.path.splitext(file_path)[1].lower()
        filename = os.path.basename(file_path)
        
        try:
            if ext == '.json':
                return self._load_json(file_path, filename)
            elif ext == '.txt':
                return self._load_text(file_path, filename)
            elif ext == '.md':
                return self._load_text(file_path, filename)
            elif ext == '.csv':
                return self._load_csv(file_path, filename)
            else:
                print(f"Unsupported file format: {ext}")
                return [], []
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            return [], []
    
    def _load_json(self, file_path: str, filename: str) -> Tuple[List[str], List[str]]:
        """Load documents from a JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = []
        doc_ids = []
        
        # Handle FAQ format
        if isinstance(data, dict) and 'faqs' in data:
            for idx, faq in enumerate(data['faqs']):
                question = faq.get('question', '')
                answer = faq.get('answer', '')
                if question and answer:
                    combined = f"Q: {question}\nA: {answer}"
                    documents.append(combined)
                    doc_ids.append(f"{filename}_faq_{idx}")
        
        # Handle list of documents
        elif isinstance(data, list):
            for idx, item in enumerate(data):
                if isinstance(item, str):
                    documents.append(item)
                    doc_ids.append(f"{filename}_{idx}")
                elif isinstance(item, dict):
                    # Handle document objects with 'content' field
                    content = item.get('content', item.get('text', str(item)))
                    documents.append(content)
                    doc_ids.append(f"{filename}_{idx}")
        
        # Handle single document object
        elif isinstance(data, dict):
            content = data.get('content', data.get('text', json.dumps(data)))
            documents.append(content)
            doc_ids.append(filename)
        
        return documents, doc_ids
    
    def _load_text(self, file_path: str, filename: str) -> Tuple[List[str], List[str]]:
        """Load documents from a text or markdown file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by paragraphs for better retrieval
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        documents = []
        doc_ids = []
        
        for idx, paragraph in enumerate(paragraphs):
            # Filter out very short paragraphs
            if len(paragraph) > 50:  # Minimum 50 characters
                documents.append(paragraph)
                doc_ids.append(f"{filename}_para_{idx}")
        
        # If no paragraphs, use the whole content
        if not documents and content.strip():
            documents.append(content)
            doc_ids.append(filename)
        
        return documents, doc_ids
    
    def _load_csv(self, file_path: str, filename: str) -> Tuple[List[str], List[str]]:
        """Load documents from a CSV file."""
        import csv
        
        documents = []
        doc_ids = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                # Combine all columns into a single document
                content = ' | '.join([f"{k}: {v}" for k, v in row.items() if v])
                if content.strip():
                    documents.append(content)
                    doc_ids.append(f"{filename}_row_{idx}")
        
        return documents, doc_ids
    
    def add_document(self, content: str, doc_id: Optional[str] = None) -> str:
        """
        Add a single document to the knowledge base.
        
        Args:
            content: The document content
            doc_id: Optional document ID (auto-generated if not provided)
            
        Returns:
            The document ID
        """
        if doc_id is None:
            doc_id = f"manual_doc_{len(os.listdir(self.knowledge_base_path))}"
        
        # Save to a text file
        os.makedirs(self.knowledge_base_path, exist_ok=True)
        file_path = os.path.join(self.knowledge_base_path, f"{doc_id}.txt")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return doc_id
    
    def load_faq_database(self, faq_path: Optional[str] = None) -> Dict[str, str]:
        """
        Load FAQ database for the FAQ system.
        
        Args:
            faq_path: Path to FAQ JSON file (default: knowledge_base/faq.json)
            
        Returns:
            Dictionary mapping questions to answers
        """
        if faq_path is None:
            faq_path = os.path.join(self.knowledge_base_path, "faq.json")
        
        if not os.path.exists(faq_path):
            return {}
        
        try:
            with open(faq_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            faq_dict = {}
            if isinstance(data, dict) and 'faqs' in data:
                for faq in data['faqs']:
                    question = faq.get('question', '')
                    answer = faq.get('answer', '')
                    if question and answer:
                        faq_dict[question] = answer
            
            return faq_dict
        except Exception as e:
            print(f"Error loading FAQ database: {e}")
            return {}
    
    def get_document_count(self) -> int:
        """Get the total number of documents in the knowledge base."""
        count = 0
        if os.path.exists(self.knowledge_base_path):
            for file_path in glob.glob(os.path.join(self.knowledge_base_path, "**/*"), recursive=True):
                if os.path.isfile(file_path):
                    ext = os.path.splitext(file_path)[1].lower()
                    if ext in self.supported_formats:
                        count += 1
        return count
    
    def get_knowledge_base_info(self) -> Dict[str, any]:
        """
        Get information about the knowledge base.
        
        Returns:
            Dictionary with knowledge base statistics
        """
        info = {
            'path': self.knowledge_base_path,
            'exists': os.path.exists(self.knowledge_base_path),
            'total_files': 0,
            'formats': {},
            'total_size_bytes': 0
        }
        
        if not os.path.exists(self.knowledge_base_path):
            return info
        
        for file_path in glob.glob(os.path.join(self.knowledge_base_path, "**/*"), recursive=True):
            if os.path.isfile(file_path):
                ext = os.path.splitext(file_path)[1].lower()
                if ext in self.supported_formats:
                    info['total_files'] += 1
                    info['formats'][ext] = info['formats'].get(ext, 0) + 1
                    info['total_size_bytes'] += os.path.getsize(file_path)
        
        return info


def load_documents_from_path(path: str) -> Tuple[List[str], List[str]]:
    """
    Convenience function to load documents from a path.
    
    Args:
        path: Path to knowledge base directory
        
    Returns:
        Tuple of (documents list, document IDs list)
    """
    ingestion = DataIngestion(path)
    return ingestion.load_all_documents()


def load_faq_from_path(path: str) -> Dict[str, str]:
    """
    Convenience function to load FAQ database from a path.
    
    Args:
        path: Path to knowledge base directory
        
    Returns:
        Dictionary mapping questions to answers
    """
    ingestion = DataIngestion(path)
    return ingestion.load_faq_database()
