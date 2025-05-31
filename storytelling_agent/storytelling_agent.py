import time
import re
import json
import os
import logging
from dotenv import load_dotenv
import openai
import google.generativeai as genai
import anthropic
from storytelling_agent import utils
from storytelling_agent.plans import plans

logger = logging.getLogger(__name__)

load_dotenv()
SUPPORTED_BACKENDS = ["openai", "gemini", "claude"]

def generate_prompt_parts(messages, include_roles=set(('user', 'assistant', 'system'))):
    last_role = None
    messages = [m for m in messages if m['role'] in include_roles]
    for idx, message in enumerate(messages):
        nl = "\n" if idx > 0 else ""
        if message['role'] == 'system':
            if idx > 0 and last_role not in (None, "system"):
                raise ValueError("system message not at start")
            yield f"{message['content']}"
        elif message['role'] == 'user':
            yield f"{nl}### USER: {message['content']}"
        elif message['role'] == 'assistant':
            yield f"{nl}### ASSISTANT: {message['content']}"
        last_role = message['role']
    if last_role != 'assistant':
        yield '\n### ASSISTANT:'

class StoryAgent:
    def __init__(self, backend_uri, backend="gemini", model=None, request_timeout=120,
                 max_tokens=4096, n_crop_previous=400, prompt_engine=None, form='novel',
                 extra_options={}, scene_extra_options={}, save_logs=False):
        self.backend = backend.lower()
        if self.backend not in SUPPORTED_BACKENDS:
            raise ValueError("Unknown backend")
        
        self.model = model 
        
        if save_logs:
            log_file = "/tmp/story_generation.log" if os.getenv("VERCEL") else "story_generation.log"
            file_handler = logging.FileHandler(log_file, mode='w')
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        else:
            logger.addHandler(logging.NullHandler())
        
        logger.info(f"Initializing StoryAgent with backend: {self.backend}, model: {self.model}")
        if self.backend == "gemini":
            self.gemini_api_key = os.getenv('GEMINI_API_KEY')
            if not self.gemini_api_key:
                raise ValueError("GEMINI_API_KEY not found in environment")
            genai.configure(api_key=self.gemini_api_key)
        elif self.backend == "openai":
            openai.api_key = os.getenv('OPENAI_API_KEY')
            if not openai.api_key:
                raise ValueError("OPENAI_API_KEY not found in environment")
        elif self.backend == "claude":
            self.claude_api_key = os.getenv('ANTHROPIC_API_KEY')
            if not self.claude_api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in environment")
            self.claude_client = anthropic.Anthropic(api_key=self.claude_api_key)

        if prompt_engine is None:
            from storytelling_agent import prompts
            self.prompt_engine = prompts
        else:
            self.prompt_engine = prompt_engine

        self.form = form
        self.max_tokens = max_tokens
        self.extra_options = extra_options
        self.scene_extra_options = extra_options.copy()
        self.scene_extra_options.update(scene_extra_options)
        self.backend_uri = backend_uri
        self.n_crop_previous = n_crop_previous
        self.request_timeout = request_timeout
        self.book_spec = None  

    def _query_chat_gemini(self, messages, retries=3, request_timeout=120, max_tokens=4096, extra_options={}):
        model_name = self.model if self.model else 'gemini-2.0-flash'
        model = genai.GenerativeModel(model_name)
        prompt = ''.join(generate_prompt_parts(messages))
        if messages and messages[-1]["role"] == "assistant":
            result_prefix = messages[-1]["content"]
        else:
            result_prefix = ''
        
        generation_config = genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=extra_options.get('temperature', 0.7),
            top_p=extra_options.get('top_p', 1.0),
        )
        
        while retries > 0:
            try:
                logger.info("Sending request to Gemini API")
                response = model.generate_content(
                    prompt,
                    generation_config=generation_config,
                    request_options={"timeout": request_timeout}
                )
                generated_text = result_prefix + response.text.strip()
                logger.info("Received response from Gemini API")
                return generated_text
            except Exception as e:
                logger.error(f"Error in Gemini API call: {e}")
                retries -= 1
                time.sleep(5)
        logger.warning("Max retries reached for Gemini API")
        return ""

    def _query_chat_openai(self, messages, retries=3, request_timeout=120, max_tokens=4096, extra_options={}):
        model_name = self.model if self.model else "gpt-4"
        while retries > 0:
            try:
                logger.info("Sending request to OpenAI API")
                response = openai.ChatCompletion.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=extra_options.get('temperature', 0.7),
                    top_p=extra_options.get('top_p', 1.0),
                    timeout=request_timeout
                )
                if messages and messages[-1]["role"] == "assistant":
                    result_prefix = messages[-1]["content"]
                else:
                    result_prefix = ''
                generated_text = result_prefix + response.choices[0].message.content.strip()
                logger.info("Received response from OpenAI API")
                return generated_text
            except Exception as e:
                logger.error(f"Error in OpenAI API call: {e}")
                retries -= 1
                time.sleep(5)
        logger.warning("Max retries reached for OpenAI API")
        return ""

    def _query_chat_claude(self, messages, retries=3, request_timeout=120, max_tokens=4096, extra_options={}):
        model_name = self.model if self.model else "claude-3-5-sonnet-20241022"
        system_message = ""
        conversation = []

        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            elif msg["role"] == "user":
                conversation.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "assistant":
                conversation.append({"role": "assistant", "content": msg["content"]})

        result_prefix = conversation[-1]["content"] if conversation and conversation[-1]["role"] == "assistant" else ""

        while retries > 0:
            try:
                logger.info("Sending request to Claude API")
                response = self.claude_client.messages.create(
                    model=model_name,
                    system=system_message,
                    messages=conversation,
                    max_tokens=max_tokens,
                    temperature=extra_options.get('temperature', 0.7),
                    top_p=extra_options.get('top_p', 1.0),
                    timeout=request_timeout
                )
                generated_text = result_prefix + response.content[0].text.strip()
                logger.info("Received response from Claude API")
                return generated_text
            except Exception as e:
                logger.error(f"Error in Claude API call: {e}")
                retries -= 1
                time.sleep(5)
        logger.warning("Max retries reached for Claude API")
        return ""

    def query_chat(self, messages, retries=3):
        if self.backend == "gemini":
            result = self._query_chat_gemini(
                messages, retries=retries, request_timeout=self.request_timeout,
                max_tokens=self.max_tokens, extra_options=self.extra_options)
        elif self.backend == "openai":
            result = self._query_chat_openai(
                messages, retries=retries, request_timeout=self.request_timeout,
                max_tokens=self.max_tokens, extra_options=self.extra_options)
        elif self.backend == "claude":
            result = self._query_chat_claude(
                messages, retries=retries, request_timeout=self.request_timeout,
                max_tokens=self.max_tokens, extra_options=self.extra_options)
        return result

    def parse_book_spec(self, text_spec):
        fields = self.prompt_engine.book_spec_fields
        spec_dict = {field: '' for field in fields}
        last_field = None
        if "\"\"\"" in text_spec[:int(len(text_spec)/2)]:
            header, sep, text_spec = text_spec.partition("\"\"\"")
        text_spec = text_spec.strip()

        for line in text_spec.split('\n'):
            pseudokey, sep, value = line.partition(':')
            pseudokey = pseudokey.lower().strip()
            matched_key = [key for key in fields
                           if (key.lower().strip() in pseudokey)
                           and (len(pseudokey) < (2 * len(key.strip())))]
            if (':' in line) and (len(matched_key) == 1):
                last_field = matched_key[0]
                if last_field in spec_dict:
                    spec_dict[last_field] += value.strip()
            elif ':' in line:
                last_field = 'other'
                spec_dict[last_field] = ''
            else:
                if last_field:
                    spec_dict[last_field] += ' ' + line.strip()
        spec_dict.pop('other', None)
        return spec_dict

    def init_book_spec(self, topic):
        logger.info("Initializing book specification")
        messages = self.prompt_engine.init_book_spec_messages(topic, self.form)
        text_spec = self.query_chat(messages)
        spec_dict = self.parse_book_spec(text_spec)

        text_spec = "\n".join(f"{key}: {value}" for key, value in spec_dict.items())
        for field in self.prompt_engine.book_spec_fields:
            while not spec_dict[field]:
                messages = self.prompt_engine.missing_book_spec_messages(field, text_spec)
                missing_part = self.query_chat(messages)
                key, sep, value = missing_part.partition(':')
                if key.lower().strip() == field.lower().strip():
                    spec_dict[field] = value.strip()
        text_spec = "\n".join(f"{key}: {value}" for key, value in spec_dict.items())
        return messages, text_spec

    def enhance_book_spec(self, book_spec):
        logger.info("Enhancing book specification")
        messages = self.prompt_engine.enhance_book_spec_messages(book_spec, self.form)
        text_spec = self.query_chat(messages)
        spec_dict_old = self.parse_book_spec(book_spec)
        spec_dict_new = self.parse_book_spec(text_spec)

        for field in self.prompt_engine.book_spec_fields:
            if not spec_dict_new[field]:
                spec_dict_new[field] = spec_dict_old[field]

        text_spec = "\n".join(f"{key}: {value}" for key, value in spec_dict_new.items())
        return messages, text_spec

    def create_plot_chapters(self, book_spec):
        logger.info("Creating plot chapters")
        messages = self.prompt_engine.create_plot_chapters_messages(book_spec, self.form)
        plan = []
        while not plan:
            text_plan = self.query_chat(messages)
            logger.info(f"Generated plot outline:\n{text_plan}")
            if text_plan:
                plan = plans.parse_text_plan(text_plan)
        return messages, plan

    def enhance_plot_chapters(self, book_spec, plan):
        logger.info("Enhancing plot chapters")
        text_plan = plans.plan_2_str(plan)
        all_messages = []
        for act_num in range(3):
            messages = self.prompt_engine.enhance_plot_chapters_messages(
                act_num, text_plan, book_spec, self.form)
            act = self.query_chat(messages)
            if act:
                act_dict = plans.parse_act(act)
                while len(act_dict['chapters']) < 2:
                    act = self.query_chat(messages)
                    act_dict = plans.parse_act(act)
                else:
                    plan[act_num] = act_dict
                text_plan = plans.plan_2_str(plan)
            all_messages.append(messages)
        return all_messages, plan

    def split_chapters_into_scenes(self, plan):
        logger.info("Splitting chapters into scenes")
        all_messages = []
        act_chapters = {}
        for i, act in enumerate(plan, start=1):
            text_act, chs = plans.act_2_str(plan, i)
            act_chapters[i] = chs
            messages = self.prompt_engine.split_chapters_into_scenes_messages(i, text_act, self.form)
            act_scenes = self.query_chat(messages)
            act['act_scenes'] = act_scenes
            all_messages.append(messages)

        for i, act in enumerate(plan, start=1):
            act_scenes = act['act_scenes']
            act_scenes = re.split(r'Chapter (\d+)', act_scenes.strip())
            act['chapter_scenes'] = {}
            chapters = [text.strip() for text in act_scenes[:] if (text and text.strip())]
            current_ch = None
            merged_chapters = {}
            for snippet in chapters:
                if snippet.isnumeric():
                    ch_num = int(snippet)
                    if ch_num != current_ch:
                        current_ch = snippet
                        merged_chapters[ch_num] = ''
                    continue
                if merged_chapters:
                    merged_chapters[ch_num] += snippet
            ch_nums = list(merged_chapters.keys()) if len(merged_chapters) <= len(act_chapters[i]) else act_chapters[i]
            merged_chapters = {ch_num: merged_chapters[ch_num] for ch_num in ch_nums}
            for ch_num, chapter in merged_chapters.items():
                scenes = re.split(r'Scene \d+.{0,10}?:', chapter)
                scenes = [text.strip() for text in scenes[1:] if (text and (len(text.split()) > 3))]
                if not scenes:
                    continue
                act['chapter_scenes'][ch_num] = scenes
        return all_messages, plan

    @staticmethod
    def prepare_scene_text(text):
        lines = text.split('\n')
        ch_ids = [i for i in range(min(5, len(lines))) if 'Chapter ' in lines[i]]
        if ch_ids:
            lines = lines[ch_ids[-1]+1:]
        sc_ids = [i for i in range(min(5, len(lines))) if 'Scene ' in lines[i]]
        if sc_ids:
            lines = lines[sc_ids[-1]+1:]

        placeholder_i = None
        for i in range(len(lines)):
            if lines[i].startswith('Chapter ') or lines[i].startswith('Scene '):
                placeholder_i = i
                break
        if placeholder_i is not None:
            lines = lines[:i]

        text = '\n'.join(lines)
        return text

    def write_a_scene(self, scene, sc_num, ch_num, plan, previous_scene=None):
        logger.info(f"Writing scene {sc_num} in chapter {ch_num}")
        text_plan = plans.plan_2_str(plan)
        messages = self.prompt_engine.scene_messages(scene, sc_num, ch_num, text_plan, self.form)
        if previous_scene:
            previous_scene = utils.keep_last_n_words(previous_scene, n=self.n_crop_previous)
            messages[1]['content'] += f'{self.prompt_engine.prev_scene_intro}\"\"\"{previous_scene}\"\"\"'
        generated_scene = self.query_chat(messages)
        generated_scene = self.prepare_scene_text(generated_scene)
        # Clean up text to format like a real novel
        generated_scene = '\n\n'.join([' '.join(p.split('\n')) for p in generated_scene.split('\n\n')])
        return messages, generated_scene

    def continue_a_scene(self, scene, sc_num, ch_num, plan, current_scene=None):
        logger.info(f"Continuing scene {sc_num} in chapter {ch_num}")
        text_plan = plans.plan_2_str(plan)
        messages = self.prompt_engine.scene_messages(scene, sc_num, ch_num, text_plan, self.form)
        if current_scene:
            current_scene = utils.keep_last_n_words(current_scene, n=self.n_crop_previous)
            messages[1]['content'] += f'{self.prompt_engine.cur_scene_intro}\"\"\"{current_scene}\"\"\"'
        generated_scene = self.query_chat(messages)
        generated_scene = self.prepare_scene_text(generated_scene)
        # Clean up text to format like a real novel
        generated_scene = '\n\n'.join([' '.join(p.split('\n')) for p in generated_scene.split('\n\n')])
        return messages, generated_scene

    def generate_story(self, topic):
        logger.info("Starting story generation")
        _, book_spec = self.init_book_spec(topic)
        self.book_spec = book_spec  
        _, book_spec = self.enhance_book_spec(book_spec)
        _, plan = self.create_plot_chapters(book_spec)
        _, plan = self.enhance_plot_chapters(book_spec, plan)
        _, plan = self.split_chapters_into_scenes(plan)

        form_text = []
        for act in plan:
            for ch_num, chapter in act['chapter_scenes'].items():
                sc_num = 1
                for scene in chapter:
                    previous_scene = form_text[-1] if form_text else None
                    _, generated_scene = self.write_a_scene(
                        scene, sc_num, ch_num, plan, previous_scene=previous_scene)
                    form_text.append(generated_scene)
                    sc_num += 1
        logger.info("Story generation completed")
        return form_text

    def get_novel_title(self):
        if not self.book_spec:
            return 'Untitled'
        lines = self.book_spec.split('\n')
        for line in lines:
            if line.startswith('Title:'):
                return line.split(':', 1)[1].strip()
        return 'Untitled'

    def generate_random_topic(self, retries=3):
        for _ in range(retries):
            messages = [
                {"role": "system", "content": "You are a creative assistant tasked with generating unique and interesting topics for novels."},
                {"role": "user", "content": "Generate a random, creative topic for a novel. Respond with only the topic, nothing else."}
            ]
            topic = self.query_chat(messages)
            if topic.strip():
                return topic.strip()
            time.sleep(5)
        raise ValueError("Failed to generate a topic after multiple attempts")