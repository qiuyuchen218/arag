import json

from scripts.batch_runner import BatchRunner


def test_resume_tracks_samples_without_qid_by_stable_original_index(tmp_path):
    runner = BatchRunner.__new__(BatchRunner)
    runner.predictions_file = tmp_path / "predictions.jsonl"
    runner.predictions_file.write_text(
        json.dumps({
            "sample_id": "sample_000001",
            "question": "second no-id question",
            "pred_answer": "done",
        }) + "\n",
        encoding="utf-8",
    )

    completed = runner._load_completed_qids()
    questions = [
        {"question": "first no-id question"},
        {"question": "second no-id question"},
        {"question": "third no-id question"},
    ]
    pending = [
        (idx, item) for idx, item in enumerate(questions)
        if runner._sample_key(item, idx) not in completed
    ]

    assert completed == {"sample_000001"}
    assert [idx for idx, _ in pending] == [0, 2]
    assert [runner._sample_key(item, idx) for idx, item in pending] == ["sample_000000", "sample_000002"]
