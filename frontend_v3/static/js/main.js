document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('uploadForm');
    const progressContainer = document.querySelector('.progress-container');
    const progressBar = document.querySelector('.progress-bar');
    const progressStatus = document.getElementById('progressStatus');
    const resultsDiv = document.getElementById('results');
    const chatForm = document.getElementById('chatForm');
    const chatInput = document.getElementById('chatInput');
    const chatMessages = document.getElementById('chatMessages');
    let currentIteration = 0;
    let currentWorkflowData = null;

    // update progress bar
    function updateProgress(status, progress, isCompleted = false, isError = false) {
        return new Promise(resolve => {
            progressBar.style.width = `${progress}%`;
            
            progressStatus.textContent = status;
            progressStatus.className = 'progress-status';
            if (isCompleted) {
                progressStatus.classList.add('completed');
            } else if (isError) {
                progressStatus.classList.add('error');
            }

            setTimeout(resolve, 500);
        });
    }

    // message to the chat
    function addChatMessage(content, isUser = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${isUser ? 'user' : 'ai'}`;
        
        const messageContent = document.createElement('div');
        messageContent.className = 'chat-message-content';
        messageContent.textContent = content;
        
        const messageTime = document.createElement('div');
        messageTime.className = 'chat-message-time';
        messageTime.textContent = new Date().toLocaleTimeString();
        
        messageDiv.appendChild(messageContent);
        messageDiv.appendChild(messageTime);
        chatMessages.appendChild(messageDiv);
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // typing indicator
    function showTypingIndicator() {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'chat-message ai typing';
        
        const messageContent = document.createElement('div');
        messageContent.className = 'chat-message-content';
        
        const typingIndicator = document.createElement('div');
        typingIndicator.className = 'typing-indicator';
        for (let i = 0; i < 3; i++) {
            typingIndicator.appendChild(document.createElement('span'));
        }
        
        messageContent.appendChild(typingIndicator);
        messageDiv.appendChild(messageContent);
        chatMessages.appendChild(messageDiv);
        
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        return messageDiv;
    }

    // remove typing indicator
    function removeTypingIndicator(indicator) {
        if (indicator && indicator.parentNode) {
            indicator.parentNode.removeChild(indicator);
        }
    }

    function updateProgressBar(progress, stage) {
        const progressBar = document.getElementById('progress-bar');
        const progressText = document.getElementById('progress-text');
        const progressContainer = document.getElementById('progress-container');
        
        if (progressBar && progressText && progressContainer) {
            progressBar.style.width = `${progress}%`;
            progressBar.setAttribute('aria-valuenow', progress);
            
            // update progress text based on stage and progress
            let stageText = '';
            switch(stage) {
                case 'starting':
                    stageText = 'Starting document processing...';
                    break;
                case 'processing_document':
                    stageText = 'Reading and processing document...';
                    break;
                case 'document_processed':
                    stageText = 'Document processed successfully, starting risk assessment...';
                    break;
                case 'running_workflow':
                    if (progress < 70) {
                        stageText = 'Running first iteration of risk assessment...';
                    } else {
                        stageText = 'Running second iteration of risk assessment...';
                    }
                    break;
                case 'formatting_results':
                    stageText = 'Risk assessment complete, formatting results...';
                    break;
                case 'saving_results':
                    stageText = 'Saving assessment results...';
                    break;
                case 'complete':
                    stageText = 'Risk assessment complete!';
                    break;
                case 'error':
                    stageText = 'Error occurred during processing';
                    break;
                default:
                    stageText = 'Processing...';
            }
            
            progressText.textContent = stageText;
            
            // show/hide progress container based on status
            if (stage === 'complete' || stage === 'error') {
                setTimeout(() => {
                    progressContainer.style.display = 'none';
                }, 2000);
            } else {
                progressContainer.style.display = 'block';
            }
        }
    }

    function checkStatus() {
        fetch('/status')
            .then(response => response.json())
            .then(data => {
                updateProgressBar(data.progress, data.stage);
                
                // checking if not complete or error
                if (data.stage !== 'complete' && data.stage !== 'error') {
                    setTimeout(checkStatus, 1000);
                } else if (data.stage === 'complete') {
                    // Reload the page to show results
                    window.location.reload();
                }
            })
            .catch(error => {
                console.error('Error checking status:', error);
                setTimeout(checkStatus, 1000);
            });
    }

    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const fileInput = document.getElementById('file');
        if (!fileInput.files.length) {
            alert('Please select a file');
            return;
        }

        const formData = new FormData();
        formData.append('file', fileInput.files[0]);

        // show progress container and hide results
        progressContainer.style.display = 'block';
        resultsDiv.style.display = 'none';
        currentIteration = 0;

        try {
            await updateProgress('Starting document processing...', 0);

            // make the API request
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            
            if (response.ok) {
                // store the workflow data for chat context
                currentWorkflowData = data;

                // document processing
                if (data.char_count) {
                    await updateProgress(`Successfully extracted ${data.char_count} characters`, 5);
                }

                // process iterations
                const maxIterations = data.iteration_count || 1;
                const progressPerIteration = 90 / maxIterations; 
                
                for (let i = 1; i <= maxIterations; i++) {
                    currentIteration = i;
                    const baseProgress = 5 + ((i - 1) * progressPerIteration);
                    
                    // start of iteration
                    await updateProgress(`Starting iteration ${i}...`, baseProgress);
                    
                    // evidence retrieval
                    await updateProgress(`Iteration ${i}: Retrieving evidence from database and APIs...`, baseProgress + (progressPerIteration * 0.2));
                    
                    // risk assessment
                    await updateProgress(`Iteration ${i}: Assessing risks across mechanistic, biomarker, endpoint, and safety domains...`, baseProgress + (progressPerIteration * 0.4));
                    
                    // de-risking
                    await updateProgress(`Iteration ${i}: Generating mitigation strategies and alternative approaches...`, baseProgress + (progressPerIteration * 0.6));
                    
                    // format review
                    await updateProgress(`Iteration ${i}: Formatting final review...`, baseProgress + (progressPerIteration * 0.8));
                    
                    // iteration complete
                    await updateProgress(`Iteration ${i} completed`, baseProgress + progressPerIteration, true);
                }

                displayResults(data);
                await updateProgress('Processing complete!', 100, true);
                
                // hide progress after a delay
                setTimeout(() => {
                    progressContainer.style.display = 'none';
                }, 2000);

                // add initial AI message to chat
                addChatMessage("I've completed the risk assessment. You can ask me questions about the results or provide feedback. What would you like to know?");
            } else {
                throw new Error(data.error || 'Failed to process document');
            }
        } catch (error) {
            progressStatus.textContent = 'Error: ' + error.message;
            progressStatus.style.color = '#dc3545';
            progressStatus.classList.add('error');
        }
    });

    // handle chat form submission
    chatForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const message = chatInput.value.trim();
        if (!message) return;

        // add user message to chat
        addChatMessage(message, true);
        chatInput.value = '';

        const typingIndicator = showTypingIndicator();

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    workflow_data: currentWorkflowData
                })
            });

            const data = await response.json();
            
            removeTypingIndicator(typingIndicator);

            if (response.ok) {
                addChatMessage(data.response);
            } else {
                throw new Error(data.error || 'Failed to get response');
            }
        } catch (error) {
            removeTypingIndicator(typingIndicator);
            
            // error message
            addChatMessage('Sorry, I encountered an error. Please try again.');
            console.error('Chat error:', error);
        }
    });

    function displayResults(data) {
        resultsDiv.style.display = 'block';
        
        // display formatted review
        document.getElementById('formattedReview').textContent = data.formatted_review;
        
        // display key recommendations in summary tab
        displayKeyRecommendations(data);
        
        // display suggestions in suggestions tab
        displaySuggestions('mechanisticSuggestions', data.mechanistic_suggestions);
        displaySuggestions('biomarkerSuggestions', data.biomarker_suggestions);
        displaySuggestions('endpointSuggestions', data.endpoint_suggestions);
        displaySuggestions('safetySuggestions', data.safety_suggestions);

        // display evidence
        const dbEvidence = data.retrieved_evidence.filter(e => e.source === 'database');
        const apiEvidence = data.retrieved_evidence.filter(e => e.source === 'api');
        
        document.getElementById('dbEvidence').innerHTML = dbEvidence.map(e => 
            `<div class="evidence-item mb-3">
                <div class="evidence-content">${e.content}</div>
            </div>`
        ).join('');
        
        document.getElementById('apiEvidence').innerHTML = apiEvidence.map(e => 
            `<div class="evidence-item mb-3">
                <div class="evidence-content">${e.content}</div>
            </div>`
        ).join('');

        // display risk assessment
        displayRiskAssessment(data.risk_assessment);
    }

    function displayKeyRecommendations(data) {
        // get the latest recommendation for each category
        const latestMechanistic = data.mechanistic_suggestions?.length ? 
            data.mechanistic_suggestions[data.mechanistic_suggestions.length - 1] : 'No mechanistic recommendations available';
        const latestBiomarker = data.biomarker_suggestions?.length ? 
            data.biomarker_suggestions[data.biomarker_suggestions.length - 1] : 'No biomarker recommendations available';
        const latestEndpoint = data.endpoint_suggestions?.length ? 
            data.endpoint_suggestions[data.endpoint_suggestions.length - 1] : 'No endpoint recommendations available';
        const latestSafety = data.safety_suggestions?.length ? 
            data.safety_suggestions[data.safety_suggestions.length - 1] : 'No safety recommendations available';

        // display each recommendation
        document.getElementById('mechanisticKeyRecommendations').innerHTML = `
            <div class="recommendation-item">
                <p>${latestMechanistic}</p>
            </div>
        `;
        document.getElementById('biomarkerKeyRecommendations').innerHTML = `
            <div class="recommendation-item">
                <p>${latestBiomarker}</p>
            </div>
        `;
        document.getElementById('endpointKeyRecommendations').innerHTML = `
            <div class="recommendation-item">
                <p>${latestEndpoint}</p>
            </div>
        `;
        document.getElementById('safetyKeyRecommendations').innerHTML = `
            <div class="recommendation-item">
                <p>${latestSafety}</p>
            </div>
        `;
    }

    function displayRiskAssessment(riskAssessment) {
        if (!riskAssessment) {
            document.getElementById('riskAssessment').textContent = 'No risk assessment available';
            return;
        }

        try {
            updateRiskBar('mechanisticRiskBar', riskAssessment.mechanistic_risk_ranking);
            updateRiskBar('biomarkerRiskBar', riskAssessment.biomarker_risk_ranking);
            updateRiskBar('endpointRiskBar', riskAssessment.endpoint_risk_ranking);
            updateRiskBar('safetyRiskBar', riskAssessment.safety_risk_ranking);

            document.getElementById('riskAssessment').innerHTML = `
                <div class="assessment-item mb-3">
                    <h6>Overall Risk Rating</h6>
                    <div class="rating-badge ${getRatingClass(riskAssessment.summary_rating)}">
                        ${riskAssessment.summary_rating}/10
                    </div>
                </div>
                <div class="assessment-item">
                    <h6>Detailed Assessment</h6>
                    <p>${riskAssessment.risk_assessment}</p>
                </div>
            `;
        } catch (error) {
            console.error('Error displaying risk assessment:', error);
            document.getElementById('riskAssessment').textContent = 'Error displaying risk assessment';
        }
    }

    function updateRiskBar(elementId, riskLevel) {
        const bar = document.getElementById(elementId);
        if (!bar) return;

        const riskMap = {
            'low': { width: '30%', color: 'bg-success' },
            'medium': { width: '60%', color: 'bg-warning' },
            'high': { width: '90%', color: 'bg-danger' }
        };

        const risk = riskMap[riskLevel.toLowerCase()] || { width: '50%', color: 'bg-secondary' };
        
        bar.style.width = risk.width;
        bar.className = `progress-bar ${risk.color}`;
        bar.textContent = riskLevel.toUpperCase();
    }

    function getRatingClass(rating) {
        if (rating <= 3) return 'rating-low';
        if (rating <= 7) return 'rating-medium';
        return 'rating-high';
    }

    function displaySuggestions(elementId, suggestions) {
        const element = document.getElementById(elementId);
        if (suggestions && suggestions.length) {
            element.innerHTML = suggestions.map(suggestion => 
                `<div class="suggestion-item">${suggestion}</div>`
            ).join('');
        } else {
            element.innerHTML = '<div class="suggestion-item">No suggestions available</div>';
        }
    }
}); 