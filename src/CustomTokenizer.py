class CustomTokenizer:
    def tokenize(self, sentence, custom_phrases=None):
        tokenized_sentence = []
        if custom_phrases is None:
            custom_phrases = {"di thang": "F", "re phai": "R", "re trai": "L", "di lui": "B"}
        while sentence:
            found_custom_phrase = False
            for custom_phrase, replacement in custom_phrases.items():
                if sentence.startswith(custom_phrase):
                    tokenized_sentence.append(replacement)
                    sentence = sentence[len(custom_phrase):].strip()
                    found_custom_phrase = True
                    break
            if not found_custom_phrase:
                tokens = sentence.split(maxsplit=1)
                tokenized_sentence.append(tokens[0])
                if len(tokens) > 1:
                    sentence = tokens[1].strip()
                else:
                    break
        return tokenized_sentence
