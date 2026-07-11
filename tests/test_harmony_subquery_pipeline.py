import math
from pathlib import Path
import sys


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import eval_locomo_comparison as compare
from eval_longmemeval_mini import LongMemExample


class FakeEmbedding:
    def embed(self, texts):
        return [self._vector(text) for text in texts]

    @staticmethod
    def _vector(text):
        lowered = text.lower()
        return [
            1.0 + lowered.count("alice"),
            1.0 + lowered.count("move") + lowered.count("moved"),
            1.0 + lowered.count("diet") + lowered.count("food"),
            1.0 + lowered.count("allergy") + lowered.count("why"),
        ]

    @staticmethod
    def cosine(left, right):
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(a * a for a in left))
        right_norm = math.sqrt(sum(b * b for b in right))
        return dot / max(1e-9, left_norm * right_norm)


class FakeReranker:
    def rerank(self, query, docs):
        terms = {term for term in query.lower().split() if len(term) > 3}
        return [sum(term in doc.lower() for term in terms) / max(1, len(terms)) for doc in docs]


def test_subquery_pipeline_uses_role_route_and_preserves_sources():
    rows = [
        {
            "row_id": "r1", "session_id": "s1", "session_index": 0, "turn_index": 0,
            "date": "2024-01-01", "role": "Alice", "has_answer": True,
            "content": "[2024-01-01] Alice: I moved to Seattle.", "raw_content": "I moved to Seattle.",
        },
        {
            "row_id": "r2", "session_id": "s1", "session_index": 0, "turn_index": 1,
            "date": "2024-01-01", "role": "Bob", "has_answer": True,
            "content": "[2024-01-01] Bob: Alice used to prefer pasta before the move.",
            "raw_content": "Alice used to prefer pasta before the move.",
        },
        {
            "row_id": "r3", "session_id": "s2", "session_index": 1, "turn_index": 0,
            "date": "2024-02-01", "role": "Alice", "has_answer": True,
            "content": "[2024-02-01] Alice: I changed my diet because of an allergy.",
            "raw_content": "I changed my diet because of an allergy.",
        },
        {
            "row_id": "r4", "session_id": "s3", "session_index": 2, "turn_index": 0,
            "date": "2024-03-01", "role": "Bob", "has_answer": False,
            "content": "[2024-03-01] Bob: I bought a new bicycle.", "raw_content": "I bought a new bicycle.",
        },
    ]
    example = LongMemExample(
        qid="synthetic", qtype="locomo_category_3",
        question="Alice moved before she changed her diet, and why did her food preference change later?",
        answer="An allergy", question_date="2024-03-01", rows=rows, answer_session_ids=["s1", "s2"],
    )
    result = compare.run_harmony_subquery(
        example,
        compare.SubqueryRouteBandit(seed=7),
        FakeEmbedding(),
        FakeReranker(),
    )
    debug = result.debug_scores[0]
    assert debug["planner"]["decomposed"]
    assert len(debug["scheduler_steps"]) >= 2
    assert any(step["route"].startswith("role_") for step in debug["scheduler_steps"])
    assert any("[source=" in fact.content for fact in result.selected_facts)
    assert {fact.metadata["row_id"] for fact in result.selected_facts} & {"r1", "r2", "r3"}
    assert result.tokens <= 1100


def test_conversation_sampling_and_split_do_not_leak_dialogues():
    examples = [
        LongMemExample(
            qid=f"dialogue_{dialogue}::qa_{question:03d}", qtype="test", question="q", answer="a",
            question_date="", rows=[], answer_session_ids=[],
        )
        for dialogue in range(4)
        for question in range(4)
    ]
    sampled = compare.conversation_balanced_sample(examples, max_examples=8, seed=7)
    assert len({compare.conversation_key(example) for example in sampled}) == 4
    train, test = compare.split_examples(sampled, train_size=4, test_size=4, seed=7, split_unit="conversation")
    assert {compare.conversation_key(example) for example in train}.isdisjoint(
        {compare.conversation_key(example) for example in test}
    )
