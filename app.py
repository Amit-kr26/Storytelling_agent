import re
import datetime
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from storytelling_agent.storytelling_agent import StoryAgent
from typing import Optional

app = FastAPI(title="AI Novel Generator")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

def sanitize_filename(filename):
    return re.sub(r'[^\w\-_\. ]', '_', filename)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/generate/", response_class=HTMLResponse)
async def generate_novel(
    request: Request,
    topic: str = Form(""),
    model: str = Form("gemini-2.0-flash"),
    temperature: float = Form(0.7),
    top_p: float = Form(1.0),
    max_tokens: int = Form(4096)
):
    try:
        # Initialize StoryAgent
        agent = StoryAgent(
            backend_uri=None,
            backend="gemini",
            model=model,
            form="novel",
            max_tokens=max_tokens,
            request_timeout=120,
            extra_options={"temperature": temperature, "top_p": top_p},
            save_logs=False 
        )

        # Generate story
        actual_topic = topic if topic else agent.generate_random_topic()
        story = agent.generate_story(actual_topic)
        novel_title = sanitize_filename(agent.get_novel_title())
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{novel_title}_{timestamp}.txt"

        # Generate novel content in memory
        novel_content = ""
        for i, scene in enumerate(story, 1):
            novel_content += f"\nScene {i}:\n{scene}\n{'-'*50}\n"

        # Pass content to template for display and download
        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "topic": actual_topic,
                "title": novel_title,
                "filename": filename,
                "model": model,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "novel_content": novel_content
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{filename}")
async def download_novel(filename: str, novel_content: Optional[str] = None):
    if not novel_content:
        raise HTTPException(status_code=404, detail="Novel content not found")
    return Response(
        content=novel_content,
        media_type="text/plain",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)