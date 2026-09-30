"""SkillZero-style general + task-specific initial skills; no evaluation labels."""
import json
import hashlib
from pathlib import Path


class SkillBank:
    def __init__(self, root, environment):
        name = next((x for x in ('alfworld', 'webshop', 'sciworld', 'sokoban') if x in environment.lower()), None)
        if name is None:
            raise ValueError(f'ETPO does not support environment {environment}')
        self.directory = Path(root).expanduser().resolve() / name
        mapping = json.loads((self.directory / 'skill_mapping.json').read_text())
        self.task_to_skill = mapping['task_to_skill']
        self.skills = {}
        for key, filename in mapping['skill_files'].items():
            path = (self.directory / filename).resolve()
            if not path.is_relative_to(self.directory):
                raise ValueError(f'Skill path escapes its directory: {filename}')
            text = path.read_text().strip()
            if not text:
                raise ValueError(f'Initial skill is empty: {path}')
            self.skills[key] = text
        if 'general_skills' not in self.skills:
            raise ValueError('Every environment must have a nonempty general skill')

    def get(self, task_type):
        sections = [self.skills['general_skills']]
        specific = self.task_to_skill.get(str(task_type))
        if specific and specific != 'general_skills':
            sections.append(self.skills[specific])
        return '\n\n'.join(sections)


    def snapshot(self, output_directory):
        """Freeze initial guidance alongside checkpoints; reject changed resume inputs."""
        payload = {'task_to_skill': self.task_to_skill, 'skills': self.skills,
                   'sha256': {k: hashlib.sha256(v.encode()).hexdigest() for k, v in self.skills.items()}}
        output = Path(output_directory).expanduser()
        output.mkdir(parents=True, exist_ok=True)
        path = output / 'etpo_initial_skills.json'
        if path.exists() and json.loads(path.read_text()) != payload:
            raise ValueError('Initial skills changed in an existing run directory; use a new run directory')
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n')
        return path
