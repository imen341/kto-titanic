import unittest
from typing import List

MIN_LETTERS_THRESHOLD: int = 7


def count_names_with_more_than_seven_letters(first_names: List[str]) -> int:
    """
    Count the number of first names with more than MIN_LETTERS_THRESHOLD letters.
    """
    count: int = 0

    for first_name in first_names:
        if len(first_name) > MIN_LETTERS_THRESHOLD:
            count += 1
            print(
                f"{first_name} est un prénom avec un nombre de lettres supérieur à "
                f"{MIN_LETTERS_THRESHOLD}"
            )
        else:
            print(
                f"{first_name} est un prénom avec un nombre de lettres inférieur ou égal à "
                f"{MIN_LETTERS_THRESHOLD}"
            )

    return count


class TestCountNamesMethod(unittest.TestCase):

    def test_count_names_with_more_than_seven_letters(self) -> None:
        first_names: List[str] = [
            "Guillaume",
            "Gilles",
            "Juliette",
            "Antoine",
            "François",
            "Cassandre",
        ]

        result: int = count_names_with_more_than_seven_letters(first_names)

        self.assertEqual(result, 4)


if __name__ == "__main__":
    unittest.main()
