import flask
from flask import Flask, request, render_template
import jinja2
import pandas as pd
import uuid
import logging
from logging import Logger
import os

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants and Configuration
TEMPLATE_DIR = 'templates'
STATIC_DIR = 'static'
RESULTS_FILE = 'user_study_results.csv'

# Exception classes
class UserStudyError(Exception):
    pass

class InvalidTaskError(UserStudyError):
    pass

# Main Class: UserStudyInterface
class UserStudyInterface:
    def __init__(self, template_dir=TEMPLATE_DIR, static_dir=STATIC_DIR, results_file=RESULTS_FILE, logger: Logger = logger):
        self.template_dir = template_dir
        self.static_dir = static_dir
        self.results_file = results_file
        self.logger = logger
        self.task_id = str(uuid.uuid4())
        self.responses = pd.DataFrame()

        # Create template and static folders if they don't exist
        if not os.path.exists(self.template_dir):
            os.makedirs(self.template_dir)
        if not os.path.exists(self.static_dir):
            os.makedirs(self.static_dir)

        # Load or create results file
        if os.path.exists(self.results_file):
            self.responses = pd.read_csv(self.results_file)
        else:
            with open(self.results_file, 'w', encoding='utf-8'): pass

    # Create comparison task function
    def create_comparison_task(self, prompt, image_set):
        if not prompt or not image_set:
            raise InvalidTaskError("Prompt or image_set cannot be empty.")
        if len(image_set) < 2:
            raise InvalidTaskError("Image set must contain at least two images.")

        self.prompt = prompt
        self.image_set = image_set
        self.logger.info(f"Created comparison task with ID: {self.task_id}")

    # Collect responses function
    def collect_responses(self):
        @app.route('/', methods=['GET', 'POST'])
        def comparison_task():
            if request.method == 'GET':
                # Generate unique response ID
                response_id = str(uuid.uuid4())
                # Render the template with task details
                return render_template(f'{self.template_dir}/comparison_task.jinja', task_id=self.task_id, response_id=response_id, prompt=self.prompt, images=self.image_set)

            elif request.method == 'POST':
                response_data = request.form
                selected_image = response_data['selected_image']
                response_id = response_data['response_id']
                self.responses = self.responses.append({'task_id': self.task_id, 'response_id': response_id, 'selected_image': selected_image}, ignore_index=True)
                self.responses.to_csv(self.results_file, index=False)
                self.logger.info(f"Collected response for task ID: {self.task_id}, response ID: {response_id}")
                return 'Response recorded!'

        # Start the Flask app
        app.run(debug=False)

    # Analyze preferences function
    def analyze_preferences(self):
        if len(self.responses) == 0:
            raise UserStudyError("No responses available for analysis.")

        # Group responses by task ID and selected image
        grouped_responses = self.responses.groupby(['task_id', 'selected_image'])
        preference_counts = grouped_responses.size().reset_index(name='count')

        # Calculate preference ratios
        total_responses_per_task = self.responses.groupby('task_id').size()
        preference_ratios = preference_counts.merge(total_responses_per_task, on='task_id')
        preference_ratios['ratio'] = preference_ratios['count'] / preference_ratios['response_id']

        self.logger.info("Preference analysis completed.")
        return preference_counts, preference_ratios

    # Export results function
    def export_results(self, filename='user_study_results.csv'):
        self.responses.to_csv(filename, index=False)
        self.logger.info(f"Results exported to {filename}")

# Helper function to create comparison tasks
def create_comparison_tasks(prompts, image_sets):
    tasks = []
    for prompt, image_set in zip(prompts, image_sets):
        task = UserStudyInterface()
        task.create_comparison_task(prompt, image_set)
        tasks.append(task)
    return tasks

# Example usage
if __name__ == '__main__':
    # Example prompts and image sets
    prompts = ["Create a diverse set of images for this prompt", "Generate multiple images with varying styles"]
    image_sets = [["image1.jpg", "image2.jpg", "image3.jpg"], ["image4.jpg", "image5.jpg", "image6.jpg"]]

    # Create comparison tasks
    comparison_tasks = create_comparison_tasks(prompts, image_sets)

    # Collect responses for all tasks
    for task in comparison_tasks:
        task.collect_responses()

    # Analyze preferences across all tasks
    preference_counts, preference_ratios = comparison_tasks[0].analyze_preferences()

    # Export results
    comparison_tasks[0].export_results()