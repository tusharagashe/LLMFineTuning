import os
from datetime import datetime
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from werkzeug.utils import secure_filename
from src.document_processor import parse_pdf_document, format_document_with_llm
from src.graph_v2 import Workflow
from langgraph.checkpoint.memory import MemorySaver
from src._constants import NEW_USER_PROMPT, LLM_CONFIGS
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import json
from bson import ObjectId
import time
from threading import Thread
from dotenv import load_dotenv
import uuid

# load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-here')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16mb max file size
app.config['HISTORY_FOLDER'] = 'history'

# ensure required directories exist
for folder in [app.config['UPLOAD_FOLDER'], app.config['HISTORY_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

# initialize chat model
chat_model = ChatOpenAI(
    model=LLM_CONFIGS["gpt-4o"]["model"],
    temperature=0.7,
    api_key=LLM_CONFIGS["gpt-4o"]["api_key"]
)

def create_workflow():
    """create a new workflow instance with fresh memory."""
    memory = MemorySaver()
    workflow = Workflow(
        name="clinical_trial_risk_assessment",
        strategy="default",
        model_name="gpt-4o",  # using gpt-4o from llm_configs
        max_iter=2
    )
    graph = workflow.build_graph(memory)
    return workflow, graph

def create_fresh_prompt():
    """create a fresh prompt template for each run."""
    return {
        "user_proposal": "",
        "mechanism": "",
        "biomarker": "",
        "endpoint": "",
        "indication": "",
        "safety": "",
        "iteration_count": 0
    }

# routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/history')
def history():
    # get list of assessment history
    history_files = []
    for filename in os.listdir(app.config['HISTORY_FOLDER']):
        if filename.endswith('.json'):
            with open(os.path.join(app.config['HISTORY_FOLDER'], filename), 'r') as f:
                data = json.load(f)
                history_files.append({
                    'id': filename.replace('.json', ''),
                    'date': data.get('date', ''),
                    'filename': data.get('filename', ''),
                    'char_count': data.get('char_count', 0)
                })
    return render_template('history.html', history=history_files)

@app.route('/analytics')
def analytics():
    # get analytics data
    analytics_data = {
        'total_assessments': len(os.listdir(app.config['HISTORY_FOLDER'])),
        'average_risk_score': calculate_average_risk_score(),
        'common_risks': get_common_risks(),
        'assessment_trends': get_assessment_trends()
    }
    return render_template('analytics.html', data=analytics_data)

@app.route('/templates')
def templates():
    return render_template('templates.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'no file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'no selected file'}), 400
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        try:
            # create a new workflow instance for this request
            workflow, graph = create_workflow()
            # parse the document
            parse_result = parse_pdf_document(filepath)
            if not parse_result["success"]:
                return jsonify({'error': 'failed to parse document'}), 400
            # create a fresh prompt for this run
            fresh_prompt = create_fresh_prompt()
            # format the document with llm
            processed_doc = format_document_with_llm(parse_result["text"], fresh_prompt)
            # generate a unique thread id using timestamp and random string
            thread_id = f"{int(time.time())}_{str(uuid.uuid4())[:8]}"
            # configure workflow
            config = {
                "configurable": {
                    "thread_id": thread_id,
                }
            }
            # run the workflow
            state = graph.invoke(processed_doc, config)
            # format the response
            retrieved_evidence = state.get('retrieved_evidence', [])
            api_messages = state.get('api_messages', [])
            # format evidence for frontend
            formatted_evidence = []
            # add database evidence
            if retrieved_evidence:
                formatted_evidence.extend([
                    {'source': 'database', 'content': evidence}
                    for evidence in retrieved_evidence
                ])
            # add api evidence
            if api_messages:
                formatted_evidence.extend([
                    {'source': 'api', 'content': msg.content}
                    for msg in api_messages
                    if hasattr(msg, 'content')
                ])
            # format risk assessment data
            risk_assessment_data = {
                'mechanistic_risk_ranking': state.get('mechanistic_risk_ranking', 'medium'),
                'biomarker_risk_ranking': state.get('biomarker_risk_ranking', 'medium'),
                'endpoint_risk_ranking': state.get('endpoint_risk_ranking', 'medium'),
                'safety_risk_ranking': state.get('safety_risk_ranking', 'medium'),
                'summary_rating': state.get('summary_rating', 5),
                'risk_assessment': state.get('risk_assessment', 'no detailed risk assessment available')
            }
            response = {
                'char_count': len(parse_result["text"]),
                'iteration_count': state.get('iteration_count', 0),
                'risk_assessment': risk_assessment_data,
                'formatted_review': state.get('final_review_paragraph', 'no formatted review available'),
                'mechanistic_suggestions': state.get('mechanistic_suggestion_history', []),
                'biomarker_suggestions': state.get('biomarker_suggestion_history', []),
                'endpoint_suggestions': state.get('endpoint_suggestion_history', []),
                'safety_suggestions': state.get('safety_suggestion_history', []),
                'retrieved_evidence': formatted_evidence
            }
            # save to history
            history_id = str(ObjectId())
            history_data = {
                'id': history_id,
                'date': datetime.now().isoformat(),
                'filename': filename,
                **response
            }
            with open(os.path.join(app.config['HISTORY_FOLDER'], f'{history_id}.json'), 'w') as f:
                json.dump(history_data, f)
            return jsonify(response)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        finally:
            # clean up the uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message')
        workflow_data = data.get('workflow_data')
        if not user_message or not workflow_data:
            return jsonify({'error': 'missing message or workflow data'}), 400
        # extract and format the workflow data
        risk_assessment = workflow_data.get('risk_assessment', 'not available')
        formatted_review = workflow_data.get('formatted_review', 'not available')
        # format suggestions with proper numbering and indentation
        mechanistic_suggestions = '\n'.join(f"{i+1}. {s}" for i, s in enumerate(workflow_data.get('mechanistic_suggestions', [])))
        biomarker_suggestions = '\n'.join(f"{i+1}. {s}" for i, s in enumerate(workflow_data.get('biomarker_suggestions', [])))
        endpoint_suggestions = '\n'.join(f"{i+1}. {s}" for i, s in enumerate(workflow_data.get('endpoint_suggestions', [])))
        safety_suggestions = '\n'.join(f"{i+1}. {s}" for i, s in enumerate(workflow_data.get('safety_suggestions', [])))
        # format evidence with source and content
        evidence = []
        for e in workflow_data.get('retrieved_evidence', []):
            source = e.get('source', 'unknown')
            content = e.get('content', '')
            if content:
                evidence.append(f"source: {source}\ncontent: {content}\n")
        # create a more focused system prompt
        system_prompt = f"""you are an expert clinical trial risk assessment assistant. you are currently discussing a specific clinical trial risk assessment that has just been completed. \n\nthe assessment results include:\n\nrisk assessment:\n{risk_assessment}\n\nformatted review:\n{formatted_review}\n\nsuggestions:\n1. mechanistic suggestions:\n{mechanistic_suggestions}\n\n2. biomarker suggestions:\n{biomarker_suggestions}\n\n3. endpoint suggestions:\n{endpoint_suggestions}\n\n4. safety suggestions:\n{safety_suggestions}\n\nsupporting evidence:\n{chr(10).join(evidence)}\n\niteration details:\n- total iterations: {workflow_data.get('iteration_count', 0)}\n- character count: {workflow_data.get('char_count', 0)}\n\nyour role is to:\n1. help users understand the risk assessment results\n2. provide guidance on next steps based on the assessment\n3. explain the implications of the findings\n4. suggest potential mitigation strategies\n5. answer specific questions about the assessment\n6. reference specific evidence and suggestions when responding\n\nguidelines for responses:\n1. always maintain context about the specific clinical trial being discussed\n2. always reference specific evidence and suggestions from the assessment\n3. if asked about next steps, focus on the specific recommendations from the assessment\n4. use the iteration details to provide context about the assessment process\n5. when discussing risks or suggestions, cite the supporting evidence\n6. if the user asks about a specific aspect (e.g., safety, biomarkers), focus on the relevant suggestions and evidence\n7. never provide generic advice - all responses must be grounded in the assessment results\n8. if you don't see specific information in the assessment results, say so instead of making assumptions\n\nremember that you are discussing a specific clinical trial risk assessment, not general life advice or other topics. your responses should be based on the actual assessment results and evidence provided above."""
        response = chat_model.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ])
        return jsonify({'response': response.content})
    except Exception as e:
        print(f"error in chat endpoint: {str(e)}")  # add logging
        return jsonify({'error': str(e)}), 500

# helper functions for analytics
def calculate_average_risk_score():
    # implement risk score calculation logic
    return 0.0

def get_common_risks():
    # implement common risks analysis
    return []

def get_assessment_trends():
    # implement assessment trends analysis
    return []

def get_recent_assessments(limit=5):
    """get the most recent assessments from the history directory."""
    try:
        history_dir = os.path.join(app.root_path, 'history')
        if not os.path.exists(history_dir):
            return []
        assessments = []
        for filename in os.listdir(history_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(history_dir, filename)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    assessments.append({
                        'id': filename.replace('.json', ''),
                        'filename': data.get('filename', filename),
                        'date': data.get('date', ''),
                        'char_count': data.get('char_count', 0)
                    })
        # sort by date (most recent first) and limit
        assessments.sort(key=lambda x: x['date'], reverse=True)
        return assessments[:limit]
    except Exception as e:
        print(f"error getting recent assessments: {str(e)}")
        return []

@app.context_processor
def inject_recent_assessments():
    """inject recent assessments into all templates."""
    return {'recent_assessments': get_recent_assessments()}

# template management
TEMPLATES_DIR = os.path.join(app.root_path, 'templates', 'assessment_templates')
os.makedirs(TEMPLATES_DIR, exist_ok=True)

@app.route('/api/templates', methods=['GET'])
def get_templates():
    """get all templates"""
    templates = []
    for category in ['phase1', 'phase2', 'phase3', 'custom']:
        category_dir = os.path.join(TEMPLATES_DIR, category)
        if os.path.exists(category_dir):
            for file in os.listdir(category_dir):
                if file.endswith('.json'):
                    template_path = os.path.join(category_dir, file)
                    with open(template_path, 'r') as f:
                        template_data = json.load(f)
                        templates.append({
                            'name': template_data.get('name', file),
                            'description': template_data.get('description', ''),
                            'category': category,
                            'created': template_data.get('created', ''),
                            'last_used': template_data.get('last_used', ''),
                            'file': file
                        })
    return jsonify(templates)

@app.route('/api/templates', methods=['POST'])
def create_template():
    """create a new template"""
    if 'file' not in request.files:
        return jsonify({'error': 'no file provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'no file selected'}), 400
    if not file.filename.endswith('.json'):
        return jsonify({'error': 'only json files are allowed'}), 400
    name = request.form.get('name')
    description = request.form.get('description', '')
    category = request.form.get('category', 'custom')
    if not name:
        return jsonify({'error': 'template name is required'}), 400
    # create category directory if it doesn't exist
    category_dir = os.path.join(TEMPLATES_DIR, category)
    os.makedirs(category_dir, exist_ok=True)
    # save template file
    filename = secure_filename(f"{name.lower().replace(' ', '_')}.json")
    file_path = os.path.join(category_dir, filename)
    template_data = {
        'name': name,
        'description': description,
        'category': category,
        'created': datetime.now().isoformat(),
        'last_used': None,
        'content': json.loads(file.read().decode('utf-8'))
    }
    with open(file_path, 'w') as f:
        json.dump(template_data, f, indent=2)
    return jsonify({'message': 'template created successfully'}), 201

@app.route('/api/templates/<category>/<filename>', methods=['GET'])
def get_template(category, filename):
    """get a specific template"""
    template_path = os.path.join(TEMPLATES_DIR, category, filename)
    if not os.path.exists(template_path):
        return jsonify({'error': 'template not found'}), 404
    with open(template_path, 'r') as f:
        template_data = json.load(f)
    return jsonify(template_data)

@app.route('/api/templates/<category>/<filename>', methods=['PUT'])
def update_template(category, filename):
    """update a template"""
    template_path = os.path.join(TEMPLATES_DIR, category, filename)
    if not os.path.exists(template_path):
        return jsonify({'error': 'template not found'}), 404
    data = request.get_json()
    if not data:
        return jsonify({'error': 'no data provided'}), 400
    with open(template_path, 'r') as f:
        template_data = json.load(f)
    template_data.update({
        'name': data.get('name', template_data['name']),
        'description': data.get('description', template_data['description']),
        'content': data.get('content', template_data['content'])
    })
    with open(template_path, 'w') as f:
        json.dump(template_data, f, indent=2)
    return jsonify({'message': 'template updated successfully'})

@app.route('/api/templates/<category>/<filename>', methods=['DELETE'])
def delete_template(category, filename):
    """delete a template"""
    template_path = os.path.join(TEMPLATES_DIR, category, filename)
    if not os.path.exists(template_path):
        return jsonify({'error': 'template not found'}), 404
    os.remove(template_path)
    return jsonify({'message': 'template deleted successfully'})

def get_workflow_status():
    """get the current status of the workflow."""
    try:
        status_file = os.path.join(app.root_path, 'workflow_status.json')
        if os.path.exists(status_file):
            with open(status_file, 'r') as f:
                return json.load(f)
        return {'stage': 'idle', 'progress': 0}
    except Exception as e:
        print(f"error reading workflow status: {str(e)}")
        return {'stage': 'idle', 'progress': 0}

def update_workflow_status(stage, progress):
    """update the workflow status."""
    try:
        status_file = os.path.join(app.root_path, 'workflow_status.json')
        with open(status_file, 'w') as f:
            json.dump({'stage': stage, 'progress': progress}, f)
    except Exception as e:
        print(f"error updating workflow status: {str(e)}")

@app.route('/process', methods=['POST'])
def process_document():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'no file provided'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'no file selected'}), 400
        # save the file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.root_path, 'uploads', filename)
        file.save(file_path)
        # initialize workflow status
        update_workflow_status('starting', 0)
        # start the workflow in a separate thread
        def run_workflow():
            try:
                # initialize memory
                memory = Memory()
                
                # initialize workflow
                workflow = Workflow(
                    strategy="default",
                    model_name="gpt-4o",
                    max_iter=2
                )
                graph = workflow.build_graph(memory)

                # process the document
                update_workflow_status('processing_document', 20)
                result = workflow.process_document(file_path, memory)
                
                # check for successful extraction
                if 'Successfully extracted' in result.get('extracted_text', ''):
                    update_workflow_status('document_processed', 40)
                
                # run the workflow and track progress
                update_workflow_status('running_workflow', 60)
                
                # track workflow iterations
                iteration_count = 0
                max_iterations = 2  # match the max_iter parameter
                
                for node in graph.nodes():
                    if node.name == 'agentic_workflow':
                        # update status for each iteration
                        iteration_count += 1
                        progress = 60 + (iteration_count / max_iterations * 20)  # 60-80% range
                        update_workflow_status('running_workflow', int(progress))
                
                workflow_result = workflow.run(graph, memory)
                
                # format and save results
                update_workflow_status('formatting_results', 80)
                formatted_result = workflow.format_results(workflow_result)
                
                # save to history
                update_workflow_status('saving_results', 90)
                history_file = os.path.join(app.root_path, 'history', f"{int(time.time())}.json")
                with open(history_file, 'w') as f:
                    json.dump(formatted_result, f)
                
                update_workflow_status('complete', 100)
                
            except Exception as e:
                print(f"error in workflow: {str(e)}")
                update_workflow_status('error', 0)

        thread = Thread(target=run_workflow)
        thread.start()

        return jsonify({
            'message': 'Processing started',
            'filename': filename
        })

    except Exception as e:
        print(f"error in process_document: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/status')
def get_status():
    """get the current status of the workflow."""
    return jsonify(get_workflow_status())

if __name__ == '__main__':
    app.run(debug=True, port=5004) 