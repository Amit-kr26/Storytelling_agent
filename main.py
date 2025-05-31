import os
import re
import datetime
from storytelling_agent.storytelling_agent import StoryAgent

def sanitize_filename(filename):
    return re.sub(r'[^\w\-_\. ]', '_', filename)

# Initialize StoryAgent with specified parameters
agent = StoryAgent(
    backend_uri=None,
    backend="gemini",
    model="gemini-2.0-flash",  # Specific model for the backend
    form="novel",
    max_tokens=4096,
    request_timeout=120,
    extra_options={"temperature": 0.7, "top_p": 1.0},
    save_logs=True  # Save logs to 'story_generation.log'
)

topic = agent.generate_random_topic()
story = agent.generate_story(topic)

# Get the AI-generated novel title
novel_title = sanitize_filename(agent.get_novel_title())
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Ensure the "novels" folder exists
if not os.path.exists("novels"):
    os.makedirs("novels")

# Save the story with the AI-generated title and timestamp in the "novels" folder
output_file = os.path.join("novels", f"{novel_title}_{timestamp}.txt")
with open(output_file, 'w') as f:
    for i, scene in enumerate(story, 1):
        f.write(f"\nScene {i}:\n{scene}\n{'-'*50}\n")

print(f"Story saved to {output_file}")