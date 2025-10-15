"""
Advanced predictive caching system with proactive generation.
Implements:
- Query prediction
- Smart caching
- Memory management
- Performance optimization
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Set
import json
import time
from dataclasses import dataclass
import threading
from collections import OrderedDict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import torch
import wandb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CacheConfig:
    """Configuration for predictive caching."""
    cache_size_mb: int = 512  # Max cache size in MB
    ttl_seconds: int = 3600  # Time-to-live for cache entries
    min_similarity: float = 0.8  # Minimum similarity for cache hit
    predict_top_k: int = 5  # Number of predictions to generate
    batch_size: int = 16
    memory_limit_gb: float = 1.0

class CacheEntry:
    """Single cache entry with metadata."""
    def __init__(
        self,
        query: str,
        response: str,
        metadata: Optional[Dict] = None
    ):
        self.query = query
        self.response = response
        self.metadata = metadata or {}
        self.created_at = time.time()
        self.hits = 0
        self._size = len(query.encode()) + len(response.encode())
        
    @property
    def size_mb(self) -> float:
        """Get entry size in MB."""
        return self._size / (1024 * 1024)
        
    @property
    def age(self) -> float:
        """Get entry age in seconds."""
        return time.time() - self.created_at
        
    def update_stats(self):
        """Update entry statistics."""
        self.hits += 1
        self.metadata["last_accessed"] = time.time()

class LRUCache:
    """LRU cache with size limits."""
    def __init__(self, max_size_mb: int = 512):
        self.max_size_mb = max_size_mb
        self.current_size_mb = 0
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.Lock()
        
    def get(self, key: str) -> Optional[CacheEntry]:
        """Get item from cache."""
        with self._lock:
            if key in self.cache:
                entry = self.cache.pop(key)
                self.cache[key] = entry  # Move to end
                entry.update_stats()
                return entry
        return None
        
    def put(self, key: str, entry: CacheEntry) -> bool:
        """Add item to cache."""
        with self._lock:
            # Check if entry would exceed size limit
            if entry.size_mb > self.max_size_mb:
                return False
                
            # Remove old items if needed
            while (self.current_size_mb + entry.size_mb > self.max_size_mb
                   and self.cache):
                _, old_entry = self.cache.popitem(last=False)
                self.current_size_mb -= old_entry.size_mb
                
            # Add new entry
            if key in self.cache:
                self.current_size_mb -= self.cache[key].size_mb
            self.cache[key] = entry
            self.current_size_mb += entry.size_mb
            return True
            
    def clear_expired(self, ttl_seconds: int):
        """Remove expired entries."""
        with self._lock:
            current_time = time.time()
            expired = [
                key for key, entry in self.cache.items()
                if current_time - entry.created_at > ttl_seconds
            ]
            for key in expired:
                entry = self.cache.pop(key)
                self.current_size_mb -= entry.size_mb

class QueryPredictor:
    """Predicts future queries using ML."""
    def __init__(self, min_similarity: float = 0.8):
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3)
        )
        self.nn = None
        self.min_similarity = min_similarity
        self.queries: List[str] = []
        
    def add_query(self, query: str):
        """Add query to training data."""
        self.queries.append(query)
        
        # Retrain if enough samples
        if len(self.queries) % 100 == 0:
            self._train()
            
    def _train(self):
        """Train prediction model."""
        try:
            # Convert queries to vectors
            X = self.vectorizer.fit_transform(self.queries)
            
            # Train nearest neighbors
            self.nn = NearestNeighbors(
                n_neighbors=5,
                metric="cosine"
            )
            self.nn.fit(X)
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            
    def predict(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Predict similar queries."""
        if not self.nn:
            return []
            
        try:
            # Convert query to vector
            query_vec = self.vectorizer.transform([query])
            
            # Find neighbors
            distances, indices = self.nn.kneighbors(
                query_vec,
                n_neighbors=k
            )
            
            # Convert to similarities
            similarities = 1 - distances.flatten()
            
            # Filter by minimum similarity
            results = [
                (self.queries[idx], sim)
                for idx, sim in zip(indices.flatten(), similarities)
                if sim >= self.min_similarity
            ]
            
            return results
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return []

class PredictiveCache:
    """Advanced predictive caching system."""
    
    def __init__(
        self,
        config: Optional[CacheConfig] = None,
        device: str = "cpu"
    ):
        self.config = config or CacheConfig()
        self.device = device
        
        # Initialize components
        self.cache = LRUCache(max_size_mb=self.config.cache_size_mb)
        self.predictor = QueryPredictor(
            min_similarity=self.config.min_similarity
        )
        
        # Background tasks
        self._setup_background_tasks()
        
        # Stats tracking
        self.stats = {
            "hits": 0,
            "misses": 0,
            "predictions": 0,
            "cache_size_mb": 0
        }
        
    def _setup_background_tasks(self):
        """Setup background maintenance tasks."""
        def maintenance_loop():
            while True:
                try:
                    # Clear expired entries
                    self.cache.clear_expired(self.config.ttl_seconds)
                    
                    # Update stats
                    self._update_stats()
                    
                    time.sleep(60)  # Run every minute
                    
                except Exception as e:
                    logger.error(f"Maintenance error: {e}")
                    time.sleep(5)
        
        self.maintenance_thread = threading.Thread(
            target=maintenance_loop,
            daemon=True
        )
        self.maintenance_thread.start()
        
    def _update_stats(self):
        """Update and log statistics."""
        self.stats["cache_size_mb"] = self.cache.current_size_mb
        
        if wandb.run is not None:
            wandb.log({
                "cache_hits": self.stats["hits"],
                "cache_misses": self.stats["misses"],
                "predictions_made": self.stats["predictions"],
                "cache_size_mb": self.stats["cache_size_mb"],
                "hit_rate": (
                    self.stats["hits"] /
                    (self.stats["hits"] + self.stats["misses"])
                    if self.stats["hits"] + self.stats["misses"] > 0
                    else 0
                )
            })
            
    def get(
        self,
        query: str,
        generate_fn: Optional[callable] = None
    ) -> Optional[str]:
        """Get response from cache or generate."""
        # Check exact match
        entry = self.cache.get(query)
        if entry:
            self.stats["hits"] += 1
            return entry.response
            
        # Check similar queries
        predictions = self.predictor.predict(
            query,
            k=self.config.predict_top_k
        )
        
        if predictions:
            self.stats["predictions"] += 1
            for pred_query, similarity in predictions:
                entry = self.cache.get(pred_query)
                if entry and similarity >= self.config.min_similarity:
                    self.stats["hits"] += 1
                    return entry.response
        
        # Cache miss
        self.stats["misses"] += 1
        
        # Generate if function provided
        if generate_fn:
            try:
                response = generate_fn(query)
                self._cache_response(query, response)
                return response
            except Exception as e:
                logger.error(f"Generation failed: {e}")
                
        return None
        
    def preload(
        self,
        queries: List[str],
        generate_fn: callable
    ) -> None:
        """Preload cache with responses."""
        try:
            # Process in batches
            for i in range(0, len(queries), self.config.batch_size):
                batch = queries[i:i + self.config.batch_size]
                
                for query in batch:
                    if not self.cache.get(query):  # Skip if cached
                        response = generate_fn(query)
                        self._cache_response(query, response)
                        
                        # Add to predictor
                        self.predictor.add_query(query)
                        
        except Exception as e:
            logger.error(f"Preload failed: {e}")
            
    def _cache_response(
        self,
        query: str,
        response: str,
        metadata: Optional[Dict] = None
    ) -> None:
        """Add response to cache."""
        try:
            entry = CacheEntry(query, response, metadata)
            self.cache.put(query, entry)
            self.predictor.add_query(query)
            
        except Exception as e:
            logger.error(f"Caching failed: {e}")
            
    def clear(self):
        """Clear cache and stats."""
        self.cache = LRUCache(max_size_mb=self.config.cache_size_mb)
        self.stats = {
            "hits": 0,
            "misses": 0,
            "predictions": 0,
            "cache_size_mb": 0
        }
        
    def get_stats(self) -> Dict[str, float]:
        """Get cache statistics."""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = (
            self.stats["hits"] / total_requests
            if total_requests > 0
            else 0
        )
        
        return {
            "hit_rate": hit_rate,
            "cache_size_mb": self.cache.current_size_mb,
            "predictions": self.stats["predictions"],
            "memory_used_gb": (
                torch.cuda.max_memory_allocated() / 1024**3
                if torch.cuda.is_available()
                else self.cache.current_size_mb / 1024
            )
        }