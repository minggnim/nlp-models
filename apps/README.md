## Instructions to run applications

1. Install the latest nlp_models package

    ```
    pip install -U nlp_models
    ```

2. Download Llama2 quantization model `llama-2-7b-chat.ggmlv3.q8_0.bin` 
    
    from https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML/tree/main and place it under `models` folder


3. Run the application of interest
    - Chat `streamlit run chat.py`
    - Q&A `streamlit run qa.py` 
