def en_to_zh():
    """
    English to chinese: https://huggingface.co/liam168/trans-opus-mt-en-zh
    """
    from transformers import AutoModelWithLMHead, AutoTokenizer, pipeline
    model_name = "liam168/trans-opus-mt-en-zh"
    model = AutoModelWithLMHead.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    translation = pipeline("translation_en_to_zh", model=model, tokenizer=tokenizer)
    # out = translation("I like to study Data Sciense and Machine Learning.", max_length=400)
    out = translation("What's your plan for Thanksgiving?", max_length=400)
    print(out)

def zh_to_en():
    """
    Chinese to English: https://huggingface.co/Helsinki-NLP/opus-mt-zh-en
    """
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
    translation = pipeline("translation_zh_to_en", model=model, tokenizer=tokenizer)
    out = translation("你感恩节有什么计划?", max_length=400)
    print(out)

# en_to_zh()
zh_to_en()
print("bye")
