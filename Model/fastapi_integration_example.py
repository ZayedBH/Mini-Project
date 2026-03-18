from fastapi import FastAPI, Request

from intent_router import route_prompt


app = FastAPI()


# Replace this with your actual coder model call.
def coder_model(prompt: str) -> str:
    return f"Model response for: {prompt}"


@app.post("/chat")
async def chat(request: Request):
    payload = await request.json()
    prompt = payload["prompt"]

    route = route_prompt(prompt)

    if route == "greeting":
        return {"response": "Hi! I can help with Python coding tasks."}

    if route == "unsupported_language":
        return {
            "response": "I can only help with Python. Please ask your coding question in Python."
        }

    if route == "out_of_scope":
        return {"response": "Out of Scope"}

    if route == "valid_intent":
        response = coder_model(prompt)
        return {"response": response}

    return {"response": "Out of Scope"}
