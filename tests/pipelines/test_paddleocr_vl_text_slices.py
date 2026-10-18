"""split_text_block_image / join_text_slices (custom-changes): python test_paddleocr_vl_text_slices.py, or pytest."""
import numpy as np

from paddlex.inference.pipelines.paddleocr_vl.uilts import join_text_slices, split_text_block_image


def page(width, height):
    return np.full((height, width, 3), 255, np.uint8)


def lines(img, n, top=10, height=20, gap=10, x0=20, x1=None, script_every=0):
    """n text lines as black bars; a 3 px superscript above every script_every-th line, 2 px above it."""
    x1 = x1 or img.shape[1] - 20
    for i in range(n):
        y = top + i * (height + gap)
        img[y:y + height, x0:x1] = 0
        if script_every and i % script_every == 0 and i:
            img[y - 5:y - 2, x0 + 40:x0 + 52] = 0
    return img


def test_tall_block_is_cut_between_lines():
    img = lines(page(900, 30 * 30 + 20), 30, script_every=4)
    parts = split_text_block_image(img, 12)
    assert len(parts) == 3, [p.shape for p in parts]
    assert sum(p.shape[0] for p in parts) == img.shape[0]
    for p in parts[1:]:
        assert (p[0] == 255).all(), "a cut went through ink"
    for p in parts:
        assert int(((p[:, :, 0] == 0).sum(axis=1) > 100).sum()) // 20 <= 12  # rows of the 20 px bars


def test_short_block_is_one_slice():
    img = lines(page(900, 12 * 30 + 20), 12)
    assert len(split_text_block_image(img, 12)) == 1


def test_off():
    img = lines(page(900, 30 * 30 + 20), 30)
    assert len(split_text_block_image(img, 0)) == 1


def test_rotated_text_is_not_cut():
    img = page(70, 900)
    for i in range(60):  # vertical text: short ink runs along a narrow column
        img[10 + i * 14:10 + i * 14 + 9, 10:30] = 0
        img[10 + i * 14:10 + i * 14 + 9, 45:62] = 0
    assert len(split_text_block_image(img, 12)) == 1


def test_two_columns_are_not_cut():
    img = page(900, 30 * 30 + 20)
    lines(img, 30, x0=20, x1=420)
    lines(img, 30, x0=480, x1=880)
    assert len(split_text_block_image(img, 12)) == 1


def test_join():
    assert join_text_slices(["a b", "c-", "d", "", " e "]) == "a b c-d e"
    assert join_text_slices(["", "x"]) == "x"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
