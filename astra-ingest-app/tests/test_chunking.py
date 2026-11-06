from app.chunking import chunk_document, chunk_text, split_long_paragraph


def test_empty_and_whitespace():
    assert chunk_text("", 1000, 150) == []
    assert chunk_text("  \n\r\n  ", 1000, 150) == []


def test_short_text_single_chunk_and_newline_normalisation():
    assert chunk_text("hello\r\nworld", 1000, 150) == ["hello\nworld"]


def test_paragraphs_are_packed_up_to_size():
    paras = ["a" * 400, "b" * 400, "c" * 400]
    chunks = chunk_text("\n\n".join(paras), 1000, 150)
    assert chunks == [paras[0] + "\n\n" + paras[1], paras[2]]
    assert all(len(c) <= 1000 for c in chunks)


def test_long_paragraph_splits_on_sentences():
    text = " ".join(f"Sentence number {i} is here." for i in range(200))
    chunks = chunk_text(text, 1000, 150)
    assert len(chunks) > 1 and all(len(c) <= 1000 for c in chunks)
    assert " ".join(chunks).split() == text.split()


def test_long_sentence_splits_on_words():
    text = ("word " * 600).strip() + ". Short one."
    chunks = split_long_paragraph(text, 100, 10)
    assert all(len(c) <= 100 for c in chunks)
    assert " ".join(chunks).split() == text.split()


def test_single_sentence_falls_back_to_overlapping_windows():
    # no sentence boundary -> fixed windows of `size` advancing by size - overlap (original behaviour)
    chunks = split_long_paragraph("x" * 2500, 1000, 150)
    assert [len(c) for c in chunks] == [1000, 1000, 800]
    text = ("word " * 600 + "end.").strip()
    windows = split_long_paragraph(text, 100, 10)
    assert all(len(c) <= 100 for c in windows) and windows[0][-10:] == windows[1][:10]


def test_chunk_document_uses_settings(make_settings):
    s = make_settings(CHUNK_SIZE=50, CHUNK_OVERLAP=5)
    chunks = chunk_document("a" * 130, s)
    assert [len(c) for c in chunks] == [50, 50, 40]
    default = make_settings()
    assert chunk_document("a" * 130, default) == ["a" * 130]
