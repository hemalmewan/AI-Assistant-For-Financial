"""
Docstring for src.services.llm_services 
This document all the AI configurations
"""

import os
from typing import Dict,Any
from transformers import AutoModelForCausalLM,BitsAndBytesConfig
from peft import PeftModel


### LLM configuration
def llm_configuration(config:Dict[str,Any]):
    """
    Docstring for llm_configuration
    
    :param config: Description
    :type config: Dict[str, Any]
    """
    llm_provider=config["llm_provider"]

    if llm_provider=="openai":
        model_name=config["model_name"]
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name,
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
            provider=llm_provider
        )
    elif llm_provider=="openrouter":
        openrouter_provider = config.get("openrouter_provider", "openai")
        openrouter_model = config.get("openrouter_model", "gpt-4o-mini")
        model_name = f"{openrouter_provider}/{openrouter_model}"
        
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            model=model_name,
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
            provider=llm_provider
        )
    elif llm_provider=="groq":
        model_name=config["model_name"]
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=model_name,
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
            provider=llm_provider
        )
    elif llm_provider=="gemini":
        model_name=config["model_name"]
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=config["model_name"],
            api_key=os.getenv("GEMINI_API_KEY"),
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
            provider=llm_provider


        )
    else:
        raise ValueError(f"Unknoen LLM provider:{llm_provider}")


## Text embedding model configuration
def text_embedding(config:Dict[str,Any]):
    """
    Docstring for text_embedding
    
    :param config: Description
    :type config: Dict[str, Any]
    """
    provider=config["text_emb_provider"]
    model_name=config["text_emb_mdoel"]

    if provider=="openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=model_name)
    
    elif provider=="cohere":
        from langchain_cohere import CohereEmbeddings
        return CohereEmbeddings(model=model_name)
    
    elif provider=="sbert":
        from langchain_community.embeddings import HuggingFaceEmbeddings
        model_kwargs = {"device": "cpu"}
        
        if config.get("normalize_embeddings", True):
            encode_kwargs = {"normalize_embeddings": True}
        else:
            encode_kwargs = {}
        
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
    
    else:
        raise ValueError(f"Unknown Text Embedding Proivder:{provider}")
    

##get finetuned LLama model
def get_finetuned_model(config:Dict[str,Any]):

    ##configure the 4-bit quatization
    bnb_config=BitsAndBytesConfig(
    load_in_4bit=config["load_in_4bit"],
    bnb_4bit_compute_dtype=config["bnb_4bit_compute_dtype"],
    bnb_4bit_quant_type=config["bnb_4bit_quant_type"],
    bnb_4bit_use_double_quant=config["bnb_4bit_use_double_quant"]
    )

    print("Loading base model.....")
    base_model_inference = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    print("Loading finetuned adapter...")
    finetuned_model = PeftModel.from_pretrained(base_model_inference, f"{config['hf_username']}/{config['hub_model_name']}")
    print("Finetuned model ready")


    return finetuned_model

