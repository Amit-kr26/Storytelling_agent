import re
import json
import logging

logger = logging.getLogger(__name__)

class plans:
    @staticmethod
    def split_by_act(original_plan):
        # First attempt: Split on bolded act headers with Roman numerals, allowing variations
        pattern = r'(\n\*\*ACT\s*[IVXLCDM]+:.*\*\*\n)'
        acts = re.split(pattern, original_plan, flags=re.IGNORECASE)
        
        # Check if the split produced at least three acts
        if len(acts) >= 7 and len(acts) % 2 == 1:  # Expecting intro + 3 pairs (header + content)
            num_acts = (len(acts) - 1) // 2
            if num_acts == 3:
                return [
                    acts[1] + acts[2],  # Act I
                    acts[3] + acts[4],  # Act II
                    acts[5] + acts[6]   # Act III
                ]
        
        # Second attempt: Fallback for simpler formats (e.g., "Act 1:", "ACT I:")
        pattern = r'(\nACT\s*[IVXLCDM]+:.*\n)'
        acts = re.split(pattern, original_plan, flags=re.IGNORECASE)
        
        if len(acts) >= 7 and len(acts) % 2 == 1:
            num_acts = (len(acts) - 1) // 2
            if num_acts == 3:
                return [
                    acts[1] + acts[2],
                    acts[3] + acts[4],
                    acts[5] + acts[6]
                ]
        
        # If both attempts fail, raise an error
        raise ValueError("Could not split text into exactly three acts")

    @staticmethod
    def parse_act(act):
        act = re.split(r'\n.{0,20}?Chapter .+:', act.strip())
        chapters = [text.strip() for text in act[1:]
                    if (text and (len(text.split()) > 3))]
        return {'act_descr': act[0].strip(), 'chapters': chapters}

    @staticmethod
    def parse_text_plan(text_plan):
        acts = plans.split_by_act(text_plan)
        if not acts:
            return []
        plan = [plans.parse_act(act) for act in acts if act]
        plan = [act for act in plan if act['chapters']]
        return plan

    @staticmethod
    def normalize_text_plan(text_plan):
        plan = plans.parse_text_plan(text_plan)
        text_plan = plans.plan_2_str(plan)
        return text_plan

    @staticmethod
    def act_2_str(plan, act_num):
        text_plan = ''
        chs = []
        ch_num = 1
        for i, act in enumerate(plan):
            act_descr = act['act_descr'] + '\n'
            if not re.search(r'Act \d', act_descr[0:50]):
                act_descr = f'Act {i+1}:\n' + act_descr
            for chapter in act['chapters']:
                if (i + 1) == act_num:
                    act_descr += f'- Chapter {ch_num}: {chapter}\n'
                    chs.append(ch_num)
                elif (i + 1) > act_num:
                    return text_plan.strip(), chs
                ch_num += 1
            text_plan += act_descr + '\n'
        return text_plan.strip(), chs

    @staticmethod
    def plan_2_str(plan):
        text_plan = ''
        ch_num = 1
        for i, act in enumerate(plan):
            act_descr = act['act_descr'] + '\n'
            if not re.search(r'Act \d', act_descr[0:50]):
                act_descr = f'Act {i+1}:\n' + act_descr
            for chapter in act['chapters']:
                act_descr += f'- Chapter {ch_num}: {chapter}\n'
                ch_num += 1
            text_plan += act_descr + '\n'
        return text_plan.strip()

    @staticmethod
    def save_plan(plan, fpath):
        with open(fpath, 'w') as fp:
            json.dump(plan, fp, indent=4)