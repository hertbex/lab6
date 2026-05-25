from __future__ import annotations

import heapq
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple

# Базовые классы из интерфейса
@dataclass(frozen=True)
class Record:
    id: int
    text: str

@dataclass(frozen=True)
class SearchResult:
    id: str
    distance: float
    score: float
    path: tuple[str, ...]
    payload: Record | None

def validate_record(raw: dict) -> Record:
    """Валидация сырых данных из JSON"""
    return Record(id=int(raw["id"]), text=str(raw["text"]))

class LeafItem:
    def __init__(self, item_id: int, vector: np.ndarray, payload: Record | None):
        self.item_id = item_id
        self.vector = vector
        self.payload = payload

class Node:
    """Узел семантического дерева"""
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.center: Optional[np.ndarray] = None
        self.radius: float = 0.0
        self.size: int = 0
        self.max_item_id: int = -1
        self.is_leaf: bool = True
        self.items: List[LeafItem] = []
        self.children: List['Node'] = []

    def update_stats(self) -> None:
        """Пересчет статистики узла: центра масс, радиуса и границ ID"""
        self.size = 0
        self.max_item_id = -1
        
        if self.is_leaf:
            self.size = len(self.items)
            if self.size > 0:
                self.max_item_id = max(item.item_id for item in self.items)
                vectors = np.array([item.vector for item in self.items])
                self.center = np.mean(vectors, axis=0)
                # Радиус - максимальное расстояние до элементов листа
                self.radius = float(np.max(np.linalg.norm(vectors - self.center, axis=1))) if self.size > 0 else 0.0
        else:
            self.size = sum(c.size for c in self.children)
            if self.size > 0:
                self.max_item_id = max(c.max_item_id for c in self.children if c.size > 0)
                centers = np.array([c.center for c in self.children if c.center is not None])
                if len(centers) > 0:
                    self.center = np.mean(centers, axis=0)
                    # Радиус должен покрывать все дочерние сферы
                    max_r = 0.0
                    for c in self.children:
                        if c.center is not None:
                            d = np.linalg.norm(self.center - c.center) + c.radius
                            if d > max_r: 
                                max_r = float(d)
                    self.radius = max_r

class SemanticBTree:
    """Семантическое дерево для O(k log N) поиска"""
    def __init__(self, t: int = 8):
        self.t = t  # максимальный размер листа
        self.root = Node("root")
        self.node_counter = 0

    def _generate_node_id(self) -> str:
        self.node_counter += 1
        return f"n{self.node_counter}"

    def insert(self, vector: np.ndarray, item_id: int, payload: Record | None = None) -> None:
        new_item = LeafItem(item_id, vector, payload)
        self._insert_recursive(self.root, new_item)

    def _insert_recursive(self, node: Node, item: LeafItem) -> None:
        if node.is_leaf:
            node.items.append(item)
            node.update_stats()
            if len(node.items) > self.t:
                self._split_leaf(node)
        else:
            # Ищем наиболее близкого ребенка
            best_child = min(node.children, key=lambda c: float(np.linalg.norm(item.vector - c.center)) if c.center is not None else 0.0)
            self._insert_recursive(best_child, item)
            node.update_stats()

    def _split_leaf(self, node: Node) -> None:
        """Разбиение листа с помощью алгоритма 2-Means"""
        items = node.items
        v1, v2 = items[0].vector, items[-1].vector
        
        # 5 итераций кластеризации для стабилизации центроидов
        for _ in range(5):
            c1, c2 = [], []
            for item in items:
                d1 = np.linalg.norm(item.vector - v1)
                d2 = np.linalg.norm(item.vector - v2)
                if d1 < d2:
                    c1.append(item)
                else:
                    c2.append(item)
            
            if not c1 or not c2:  # Fallback
                mid = len(items) // 2
                c1, c2 = items[:mid], items[mid:]
                
            v1 = np.mean([i.vector for i in c1], axis=0)
            v2 = np.mean([i.vector for i in c2], axis=0)

        child1 = Node(self._generate_node_id())
        child1.is_leaf = True
        child1.items = c1
        child1.update_stats()

        child2 = Node(self._generate_node_id())
        child2.is_leaf = True
        child2.items = c2
        child2.update_stats()

        node.is_leaf = False
        node.items = []
        node.children = [child1, child2]
        node.update_stats()

    def search(self, query: np.ndarray, k: int = 5) -> List[SearchResult]:
        """Поиск k ближайших соседей (Best-First Search)"""
        pq: List[Tuple[float, int, int, Node | LeafItem, Tuple[str, ...]]] = []
        
        if self.root.size == 0:
            return []
            
        # Формат очереди: (расстояние, тип (1-узел, 0-элемент), id_объекта, объект, путь)
        heapq.heappush(pq, (0.0, 1, id(self.root), self.root, (self.root.node_id,)))
        
        results = []
        visited_items = set()
        
        while pq and len(results) < k:
            dist, type_flag, _, obj, path = heapq.heappop(pq)
            
            if type_flag == 0: # Дошли до конкретного элемента
                if obj.item_id not in visited_items:
                    visited_items.add(obj.item_id)
                    score = 1.0 - (dist**2) / 2.0  # Конвертация Euclidean в Cosine Similarity
                    results.append(SearchResult(
                        id=str(obj.item_id),
                        distance=float(dist),
                        score=float(score),
                        path=path,
                        payload=obj.payload
                    ))
            else: # Исследуем узел
                node: Node = obj # type: ignore
                if node.is_leaf:
                    for item in node.items:
                        d = float(np.linalg.norm(query - item.vector))
                        heapq.heappush(pq, (d, 0, item.item_id, item, path))
                else:
                    for child in node.children:
                        if child.center is not None:
                            # Эвристика: расстояние до сферы (расстояние до центра минус радиус)
                            d = max(0.0, float(np.linalg.norm(query - child.center)) - child.radius)
                            heapq.heappush(pq, (d, 1, id(child), child, path + (child.node_id,)))
                            
        return results

def build_tree_from_records(records: List[Record], embeddings: any, t: int = 8) -> SemanticBTree: # type: ignore
    """Утилита сборки дерева, используемая в st_demo.py"""
    tree = SemanticBTree(t=t)
    texts = [r.text for r in records]
    
    if hasattr(embeddings, 'embed_batch') and texts:
        vecs = embeddings.embed_batch(texts)
        for r, v in zip(records, vecs):
            tree.insert(v, r.id, r)
    else:
        for r in records:
            v = embeddings.embed(r.text)
            tree.insert(v, r.id, r)
            
    return tree

def linear_search(
    vectors: np.ndarray,
    payloads: list[Record],
    query: np.ndarray,
    k: int,
) -> list[SearchResult]:
    """Наивный O(k*N) поиск с использованием numpy векторизации"""
    distances = np.linalg.norm(vectors - query, axis=1)
    
    # Эффективный поиск top-k индексов
    k = min(k, len(vectors))
    nearest_idx = np.argpartition(distances, k - 1)[:k]
    nearest_idx = nearest_idx[np.argsort(distances[nearest_idx])]
    
    results = []
    for idx in nearest_idx:
        dist = distances[idx]
        score = 1.0 - (dist**2) / 2.0
        results.append(SearchResult(
            id=str(payloads[idx].id),
            distance=float(dist),
            score=float(score),
            path=("linear_scan",),
            payload=payloads[idx]
        ))
    return results


# Блок бенчмаркинга
if __name__ == "__main__":
    import time
    
    print("Генерация тестовых данных (размерность 384)...")
    dim = 384
    np.random.seed(42)
    k = 10
    
    for N in [100, 500, 1000, 5000, 10000]:
        # Эмулируем эмбеддинги
        raw_vecs = np.random.randn(N, dim).astype(np.float32)
        vectors = raw_vecs / np.linalg.norm(raw_vecs, axis=1, keepdims=True)
        
        records = [Record(id=i, text=f"mock_{i}") for i in range(N)]
        query = np.random.randn(dim).astype(np.float32)
        query = query / np.linalg.norm(query)
        
        # Замер линейного поиска
        start_lin = time.perf_counter()
        linear_search(vectors, records, query, k)
        time_lin = (time.perf_counter() - start_lin) * 1000
        
        # Сборка дерева и замер
        tree = SemanticBTree(t=16)
        for i, (v, r) in enumerate(zip(vectors, records)):
            tree.insert(v, r.id, r)
            
        start_tree = time.perf_counter()
        tree.search(query, k)
        time_tree = (time.perf_counter() - start_tree) * 1000
        
        print(f"N={N:<6} | Линейный: {time_lin:>6.2f} ms | Дерево: {time_tree:>6.2f} ms")