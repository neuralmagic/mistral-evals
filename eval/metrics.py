import re
import string

def _normalize_string(s):
    if (s.startswith('"') and s.endswith('"')) or (
        s.startswith("'") and s.endswith("'")
    ):
        return s[1:-1]
    return s


def _remove_end_punctuation(unnormalized_string: str) -> str:
    while (
        unnormalized_string
        and (
            unnormalized_string[-1] in string.punctuation
            or unnormalized_string[-1].isspace()
        )
        and unnormalized_string[-1] != "%"
    ):
        unnormalized_string = unnormalized_string[:-1]
    return unnormalized_string


class Metric:
    """Base class for metrics."""

    @property
    def name(self) -> str:
        raise NotImplementedError

    def score(self, model_answer: str, reference_answer: str | list[str]) -> float:
        raise NotImplementedError


class VQAMatch(Metric):
    """VQA match metric which gives partial score if less than 3 answers are matched."""

    def __init__(self):
        self.contractions = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't", \
                            "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't", \
                            "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd", "hed've": "he'd've", \
                            "he'dve": "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", \
                            "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's", \
                            "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've", \
                            "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't", \
                            "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've", \
                            "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've", \
                            "somebody'd": "somebodyd", "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll": "somebody'll", \
                            "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've", "someone'dve": "someone'd've", \
                            "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've", \
                            "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's", "thered": "there'd", "thered've": "there'd've", \
                            "there'dve": "there'd've", "therere": "there're", "theres": "there's", "theyd": "they'd", "theyd've": "they'd've", \
                            "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've", "twas": "'twas", "wasnt": "wasn't", \
                            "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't", "whatll": "what'll", "whatre": "what're", \
                            "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd", "wheres": "where's", "whereve": "where've", \
                            "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll", \
                            "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've", \
                            "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've", \
                            "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've", \
                            "youll": "you'll", "youre": "you're", "youve": "you've"}
        self.articles     = ['a',
                            'an',
                            'the'
                            ]

        self.periodStrip = re.compile(r'(?<!\d)\.(?!\d)')  # Matches periods not part of decimal numbers
        self.commaStrip = re.compile(r"(\d)(\,)(\d)")
        self.punct = [';', r"/", '[', ']', '"', '{', '}',
                      '(', ')', '=', '+', '\\', '_', '-',
                      '>', '<', '@', '`', ',', '?', '!']


    @property
    def name(self) -> str:
        return "vqa_match"
    
    def clean_text(self, text: str) -> str:
        """
        Cleans input text by:
        1. Removing punctuation except periods in decimal numbers.
        2. Removing commas in digit sequences.
        3. Replacing contractions.
        4. Removing articles.
        """
        # Remove commas in digit sequences (e.g., 1,000 -> 1000)
        text = self.commaStrip.sub(r'\1\3', text)

        # Replace punctuations with spaces or remove them
        for p in self.punct:
            if (p + ' ' in text or ' ' + p in text) or re.search(self.commaStrip, text):
                text = text.replace(p, '')
            else:
                text = text.replace(p, ' ')

        # Remove periods that are not part of decimal numbers
        text = self.periodStrip.sub('', text)

        # Handle contractions and remove articles
        words = text.split()
        cleaned_words = []
        for word in words:
            # Replace contractions
            word = self.contractions[word] if word in self.contractions else word
            # Exclude articles
            if word not in self.articles:
                cleaned_words.append(word)

        return ' '.join(cleaned_words).strip()

    def score(self, model_answer: str, reference_answer: str | list[str]) -> float:
        # Ensure reference_answer is a list
        if not isinstance(reference_answer, list):
            reference_answer = [reference_answer]

        # Normalize the model answer
        normalized_model_answer = _normalize_string(model_answer)

        # Clean model answer
        cleaned_model_answer = self.clean_text(
            normalized_model_answer.replace('\n', ' ').replace('\t', ' ').strip().lower()
        )

        # Precompute normalized reference answers
        normalized_answers = []
        for ans in reference_answer:
            ans_text = ans['answer'] if isinstance(ans, dict) else ans
            cleaned_ans = self.clean_text(
                ans_text.replace('\n', ' ').replace('\t', ' ').strip().lower()
            )
            normalized_answers.append(_normalize_string(cleaned_ans))
        
        # Find all matching answers (model answer vs reference answers)
        matching_answers = [
            answer for answer in normalized_answers if cleaned_model_answer == answer
        ]
        # Return the score based on matches (clamped to a max of 1.0)
        return min(1.0, float(len(matching_answers)) / 3)

class ANLS(Metric):
    @property
    def name(self) -> str:
        return "anls"

    def _edit_distance_helper(self, s1: str, s2: str) -> float:
        if len(s1) > len(s2):
            s1, s2 = s2, s1
        distances = list(range(len(s1) + 1))
        for i2, c2 in enumerate(s2):
            distance_list = [i2 + 1]
            for i1, c1 in enumerate(s1):
                if c1 == c2:
                    distance_list.append(distances[i1])
                else:
                    distance_list.append(
                        1 + min((distances[i1], distances[i1 + 1], distance_list[-1]))
                    )
            distances = distance_list
        return distances[-1]

    def score(self, model_answer: str, reference_answer: str | list[str]) -> float:
        if not isinstance(reference_answer, list):
            reference_answer = [reference_answer]

        model_answer = " ".join(model_answer.strip().lower().split())
        model_answer = _remove_end_punctuation(model_answer)

        min_value = float("inf")
        for ref in reference_answer:
            # Post-processing: Normalize spaces and remove punctuations.
            ref = " ".join(ref.strip().lower().split())
            ref = _remove_end_punctuation(ref)

            # Compute edit distance
            dist = self._edit_distance_helper(ref, model_answer)
            length = max(len(ref), len(model_answer))
            value = 0.0 if length == 0 else float(dist) / float(length)
            if value < min_value:
                min_value = value

        anls_threshold = 0.0
        output = 0.0 if 1 - min_value < anls_threshold else 1 - min_value
        return output


class RelaxedCorrectness(Metric):
    """Relaxed correctness metrics.

    The correctness tolerates certain error ratio defined by max_relative_change.
    See https://arxiv.org/pdf/2203.10244.pdf, end of section 5.1:
    "Following Methani et al. (2020), we use a relaxed accuracy measure for the
    numeric answers to allow a minor inaccuracy that may result from the automatic
    data extraction process. We consider an answer to be correct if it is within
    5% of the gold answer. For non-numeric answers, we still need an exact match
    to consider an answer to be correct."
    """

    def _relaxed_correctness(
        self, prediction: str, targets: list[str], max_relative_change: float = 0.05
    ) -> float:
        def _to_float(text: str) -> tuple[float | None, bool]:
            text = text.strip()
            is_percent = text.endswith("%")
            try:
                value = float(text.rstrip("%"))
                return value, is_percent
            except ValueError:
                return None, False

        def _is_letter(text: str) -> bool:
            return text.isalpha() and len(text) == 1

        def _preprocess_text(text: str) -> str:
            if not any(char.isdigit() for char in text):
                return _normalize_string(text)
            else:
                return _remove_end_punctuation(text).replace(",", "").replace("$", "")

        def calculate_relative_change(prediction: float, target: float) -> float:
            return abs(prediction - target) / max(abs(target), 1e-10)

        def _compare_numeric_values(
            prediction: float, target: float, max_relative_change: float
        ) -> float:
            relative_change = calculate_relative_change(prediction, target)
            return 1.0 if relative_change <= max_relative_change else 0.0

        def _compare_text_values(prediction: str, target: str) -> float:
            return 1.0 if prediction.lower() == target.lower() else 0.0

        def _to_decimal(value: float, is_percent: bool) -> float:
            return value / 100 if is_percent else value

        def _compare_numeric_with_percent(
            prediction: float,
            prediction_is_percent: bool,
            target: float,
            target_is_percent: bool,
            max_relative_change: float,
        ) -> float:
            # Compare as-is
            value = _compare_numeric_values(prediction, target, max_relative_change)

            # If not equal and one is percent, try other comparisons
            if value != 1.0 and (prediction_is_percent or target_is_percent):
                value = max(
                    value,
                    _compare_numeric_values(
                        _to_decimal(prediction, prediction_is_percent),
                        target,
                        max_relative_change,
                    ),
                    _compare_numeric_values(
                        prediction,
                        _to_decimal(target, target_is_percent),
                        max_relative_change,
                    ),
                )
            return value

        prediction = _preprocess_text(prediction)
        prediction_float, prediction_is_percent = _to_float(prediction)

        value_list = []
        for target in targets:
            target = _preprocess_text(target)
            target_float, target_is_percent = _to_float(target)

            if prediction_float is not None and target_float is not None:
                # Compare as numeric values
                value = _compare_numeric_with_percent(
                    prediction_float,
                    prediction_is_percent,
                    target_float,
                    target_is_percent,
                    max_relative_change,
                )
            elif _is_letter(target) and len(prediction) > 0:
                # Compare as multiple choice options: take first letter from prediction
                value = 1.0 if prediction[0].lower() == target.lower() else 0.0
            else:
                # Compare as text values
                value = _compare_text_values(prediction, target)

            value_list.append(value)

        return max(value_list)

    def score(self, model_answer: str, reference_answer: str | list[str]) -> float:
        reference_answer = (
            reference_answer
            if isinstance(reference_answer, list)
            else [reference_answer]
        )
        return self._relaxed_correctness(model_answer, reference_answer)


class ExplicitPromptRelaxedCorrectness(RelaxedCorrectness):
    """Relaxed correctness for explicit prompt."""

    @property
    def name(self) -> str:
        return "explicit_prompt_relaxed_correctness"

    def _get_final_answer(self, generation: str) -> str:
        def _find_last_occurrence(pattern: str, string: str):
            return string.rfind(pattern)

        # Strip extraneous markdown around the answer:
        generation = re.sub(r"([aA]nswer)\**:\**", "\\1:", generation)

        final_answer_index = _find_last_occurrence("answer:", generation.lower())

        if final_answer_index != -1:
            # Find the start of the answer (after "final answer:")
            start_index = final_answer_index + len("answer:")

            # Split the remaining text into lines
            lines = generation[start_index:].split("\n")

            # Find the first non-empty line
            final_answer = next((line.strip() for line in lines if line.strip()), "")

            # Remove any markdown formatting
            final_answer = re.sub(r"[*_\[\]\(\)]", "", final_answer)

            return final_answer
        else:
            return ""

    def score(self, model_answer: str, reference_answer: str | list[str]) -> float:
        parsed_model_answer = self._get_final_answer(model_answer)
        if not parsed_model_answer:
            # Parsing failed.
            return 0.0
        return super().score(parsed_model_answer, reference_answer)


class AnywhereInAnswerRelaxedCorrectness(ExplicitPromptRelaxedCorrectness):
    """Falls back to handle cases where reference answer appears anywhere in generation.

    NOTE: This is an overly generous metric and is likely to falsely inflate scores.
    """

    @property
    def name(self) -> str:
        return "anywhere_in_answer_relaxed_correctness"

    def score(self, model_answer: str, reference_answer: str | list[str]) -> float:
        reference_answer = (
            reference_answer
            if isinstance(reference_answer, list)
            else [reference_answer]
        )
        parsed_model_answer = self._get_final_answer(model_answer)
        if parsed_model_answer:
            return self._relaxed_correctness(parsed_model_answer, reference_answer)

        # Fallback: check if reference answer appears anywhere in the model answer.
        for ref in reference_answer:
            try:
                # Try to parse as a float
                number = float(ref)

                # Revert to int if it is actually an int.
                if int(number) == number:
                    number = int(number)
                # Check if the number is in the model answer with commas (e.g. 1,000)
                if format(number, ",") in model_answer:
                    return 1.0
                # Check if the number is in the model answer without commas (e.g. 1000)
                elif str(number) in model_answer:
                    return 1.0
                elif str(number) + "%" in model_answer:
                    return 1.0
            except ValueError:
                # Reference answer was a text string. We search for typical patterns
                # in the model answer. Note that directly searching for the reference
                # is not a good idea for letter-option choice questions, hence we look
                # for common patterns. This is still heuristic, and might have false
                # positives as well as false negatives.
                candidates = []
                for ref in reference_answer:
                    candidates.extend(
                        [
                            f"is {ref}",
                            f"was {ref}",
                            f" {ref}.",
                            f"are {ref}",
                            f"\n\n{ref}",
                        ]
                    )
                if any([c.lower() in model_answer for c in candidates]):
                    return 1.0

        return 0
