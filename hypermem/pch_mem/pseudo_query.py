"""
Query-free pseudo-query generation for cold-start initialization.

Generates self-supervised retrieval tasks from the structural hypergraph
without using test QA pairs, enabling policy edge initialization
for new conversations.

Reference: Paper Section 4.3.1
"""

from __future__ import annotations

import hashlib
import re
from typing import Dict, List, Set, Tuple

import numpy as np

from .types import HypergraphState, PCHConfig


class PseudoQueryGenerator:
    """Generates pseudo-queries from structural hypergraph content.

    Task types:
    - Attribute localization: "What did [role] say about [topic]?"
    - Temporal comparison: "How did [topic] change before/after [event]?"
    - Episode causality: "Why did [event] happen?"
    - Cross-role evidence: "What did [role1] and [role2] discuss about [topic]?"
    - Multi-fact aggregation: "Summarize [topic]"
    """

    TEMPLATES = {
        "attribute_lookup": [
            "What did {role} say about {topic}?",
            "When did {role} mention {topic}?",
            "Where was {role} during {episode}?",
        ],
        "temporal_comparison": [
            "How did {topic} change after {episode}?",
            "What happened before {episode}?",
            "Compare {topic} before and after {time_point}.",
        ],
        "episode_causality": [
            "Why did {episode} happen?",
            "What caused the change in {topic}?",
            "What led to {episode}?",
        ],
        "cross_role": [
            "What did {role1} and {role2} discuss about {topic}?",
            "How did {role1}'s view on {topic} differ from {role2}?",
        ],
        "multi_fact": [
            "Summarize what happened regarding {topic}.",
            "What are the key facts about {episode}?",
            "List everything mentioned about {topic}.",
        ],
    }

    def __init__(self, config: PCHConfig):
        self.config = config

    def generate(
        self,
        hypergraph: HypergraphState,
    ) -> List[Tuple[str, str, np.ndarray]]:
        """Generate pseudo-queries from structural hypergraph.

        Returns:
            List of (query_id, query_text, query_embedding) tuples.
        """
        queries: List[Tuple[str, str, np.ndarray]] = []
        generated_texts: Set[str] = set()

        # Extract entities from hypergraph
        roles = self._extract_roles(hypergraph)
        topics = self._extract_topics(hypergraph)
        episodes = self._extract_episodes(hypergraph)

        if not roles:
            roles = ["the user"]
        if not topics:
            topics = ["the conversation"]
        if not episodes:
            episodes = ["a recent event"]

        # Generate for each topic
        for topic in topics[:10]:
            for _ in range(self.config.pseudo_queries_per_topic):
                role = roles[np.random.randint(0, len(roles))]
                episode = episodes[np.random.randint(0, len(episodes))]

                # Pick a random template type
                template_types = list(self.TEMPLATES.keys())
                ttype = template_types[np.random.randint(0, len(template_types))]
                templates = self.TEMPLATES[ttype]
                template = templates[np.random.randint(0, len(templates))]

                query_text = template.format(
                    role=role,
                    topic=topic,
                    episode=episode,
                    role1=role,
                    role2=roles[np.random.randint(0, len(roles))],
                    time_point="the beginning",
                )

                if query_text in generated_texts:
                    continue
                generated_texts.add(query_text)

                query_id = f"pseudo_{hashlib.md5(query_text.encode()).hexdigest()[:8]}"
                embedding = self._embed_text(query_text)
                queries.append((query_id, query_text, embedding))

                if len(queries) >= self.config.max_pseudo_queries:
                    break

            if len(queries) >= self.config.max_pseudo_queries:
                break

        # Generate for each episode
        for episode in episodes[:10]:
            for _ in range(self.config.pseudo_queries_per_episode):
                topic = topics[np.random.randint(0, len(topics))]

                query_text = f"What happened during {episode}?"
                if query_text in generated_texts:
                    query_text = f"Tell me about {episode} in the context of {topic}."
                if query_text in generated_texts:
                    continue

                generated_texts.add(query_text)
                query_id = f"pseudo_{hashlib.md5(query_text.encode()).hexdigest()[:8]}"
                embedding = self._embed_text(query_text)
                queries.append((query_id, query_text, embedding))

                if len(queries) >= self.config.max_pseudo_queries:
                    break

            if len(queries) >= self.config.max_pseudo_queries:
                break

        return queries

    def _extract_roles(self, hg: HypergraphState) -> List[str]:
        """Extract unique roles from structural edges."""
        roles: Set[str] = set()
        for edge in hg.structural_edges.values():
            for role in edge.role_constraints.values():
                if role and role.strip():
                    roles.add(role.strip())
        return list(roles) if roles else ["user", "assistant"]

    def _extract_topics(self, hg: HypergraphState) -> List[str]:
        """Extract topic keywords from structural edges."""
        keywords: Set[str] = set()
        for edge in hg.structural_edges.values():
            for kw in edge.keywords:
                if kw and kw.strip():
                    keywords.add(kw.strip())
        # Also use fact content as topic
        for i, content in enumerate(hg.fact_contents.values()):
            if i > 20:
                break
            # Extract noun phrases (simple heuristic)
            words = content.split()
            for j in range(len(words) - 1):
                bigram = f"{words[j]} {words[j+1]}"
                if len(bigram) > 8 and len(bigram) < 40:
                    keywords.add(bigram)
        return list(keywords)[:20] if keywords else ["general discussion"]

    def _extract_episodes(self, hg: HypergraphState) -> List[str]:
        """Extract episode descriptions."""
        episodes: Set[str] = set()
        for edge in hg.structural_edges.values():
            if edge.temporal_range:
                episodes.add(f"events around {edge.temporal_range[0]}")
            if edge.keywords:
                episodes.add(f"the {edge.keywords[0]} discussion")
        return list(episodes)[:15] if episodes else ["the conversation"]

    def _embed_text(self, text: str) -> np.ndarray:
        """Generate a simple hash-based embedding for pseudo-queries.

        In production, this would use the same embedding model
        as the rest of the system (e.g., Qwen3-Embedding-4B).
        """
        dim = 256
        vec = np.zeros(dim, dtype=np.float32)
        tokens = re.findall(r"[a-z0-9_]+|[一-鿿]", text.lower())
        for tok in tokens:
            digest = hashlib.md5(tok.encode("utf-8")).hexdigest()
            idx = int(digest[:8], 16) % dim
            sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
            vec[idx] += sign
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec


def generate_pseudo_queries(
    hypergraph: HypergraphState,
    config: PCHConfig,
) -> List[Tuple[str, str, np.ndarray]]:
    """Convenience function to generate pseudo-queries."""
    generator = PseudoQueryGenerator(config)
    return generator.generate(hypergraph)
