import pytest


def get_grade_from_mark(mark):
    if mark > 50:
        return "Pass"
    else:
        return "Fail"

@pytest.mark.parametrize('mark', [65, 80, 50])
def test_get_grade_pass(mark):
    grade = get_grade_from_mark(mark)
    assert grade == "Pass", f"Expected {mark} to pass, but result was {grade}"

def test_get_grade_fail():
    grade = get_grade_from_mark(43)
    assert grade == "Fail", f"Expected 43 to fail, but result was {grade}"    

if __name__ == "__main__":
    test_get_grade_pass(65)
    test_get_grade_fail(43)
    print("All passing.")
