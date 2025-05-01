// Main app functionality
document.addEventListener('DOMContentLoaded', function() {
    // Generate a session ID
    const sessionId = generateSessionId();
    
    // DOM elements
    const messagesContainer = document.getElementById('messagesContainer');
    const userInput = document.getElementById('userInput');
    const sendButton = document.getElementById('sendButton');
    const calculatorSelect = document.getElementById('calculatorSelect');
    const calculatorUrlContainer = document.getElementById('calculatorUrlContainer');
    const calculatorUrl = document.getElementById('calculatorUrl');
    const useCalculatorBtn = document.getElementById('useCalculatorBtn');
    const modelSelect = document.getElementById('modelSelect');
    const newConversationBtn = document.getElementById('newConversationBtn');
    const saveConversationBtn = document.getElementById('saveConversationBtn');
    const savedConversationsList = document.getElementById('savedConversationsList');
    
    // State
    let isTyping = false;
    let websocket = null;
    let selectedCalculator = null;
    let selectedCalculatorUrl = null;
    
    // Initialize the application
    init();
    
    // Main initialization function
    function init() {
        // Load available calculators
        loadCalculators();
        
        // Load saved conversations
        loadSavedConversations();
        
        // Connect WebSocket
        connectWebSocket();
        
        // Set up event listeners
        setupEventListeners();
    }
    
    // Connect to WebSocket
    function connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/${sessionId}`;
        
        websocket = new WebSocket(wsUrl);
        
        websocket.onopen = function(e) {
            console.log('WebSocket connection established');
        };
        
        websocket.onmessage = function(event) {
            handleWebSocketMessage(JSON.parse(event.data));
        };
        
        websocket.onclose = function(e) {
            console.log('WebSocket connection closed. Reconnecting in 1s...');
            // Try to reconnect after a delay
            setTimeout(connectWebSocket, 1000);
        };
        
        websocket.onerror = function(error) {
            console.error('WebSocket error:', error);
        };
    }
    
    // Handle WebSocket messages
    function handleWebSocketMessage(data) {
        console.log('Received WebSocket message:', data);
        
        switch(data.type) {
            case 'history':
                // Load chat history
                renderMessages(data.data.messages);
                if (data.data.selected_calculator) {
                    // Update calculator selection
                    selectedCalculator = data.data.selected_calculator;
                    selectedCalculatorUrl = data.data.calculator_url;
                    updateCalculatorUI();
                }
                break;
                
            case 'message':
                // Remove typing indicator
                removeTypingIndicator();
                
                // Add message to chat
                addMessageToChat(data.data.role, data.data.content);
                break;
                
            case 'message_received':
                // Acknowledgment of message received
                break;
                
            case 'calculator_selected':
                // Calculator was selected
                selectedCalculator = data.data.calculator;
                addMessageToChat('assistant', data.data.message);
                break;
                
            case 'new_conversation':
                // New conversation started
                messagesContainer.innerHTML = '';
                renderMessages(data.data.messages);
                selectedCalculator = null;
                selectedCalculatorUrl = null;
                updateCalculatorUI();
                break;
                
            case 'conversation_saved':
                showNotification('Conversation saved successfully!');
                break;
                
            case 'model_updated':
                showNotification('Model updated to ' + data.data.model);
                break;
                
            case 'error':
                showNotification(data.data, 'error');
                break;
                
            default:
                console.log('Unknown message type:', data.type);
        }
    }
    
    // Load available calculators
    async function loadCalculators() {
        try {
            const response = await fetch('/api/calculators');
            const data = await response.json();
            
            // Clear existing options
            calculatorSelect.innerHTML = '<option value="" selected disabled>Choose a calculator...</option>';
            
            // Add calculator options
            for (const [name, url] of Object.entries(data)) {
                const option = document.createElement('option');
                option.value = name;
                option.textContent = name;
                option.dataset.url = url;
                calculatorSelect.appendChild(option);
            }
        } catch (error) {
            console.error('Error loading calculators:', error);
            showNotification('Failed to load calculators', 'error');
        }
    }
    
    // Load saved conversations
    async function loadSavedConversations() {
        try {
            const response = await fetch('/api/conversations');
            const data = await response.json();
            
            // Clear the list
            savedConversationsList.innerHTML = '';
            
            if (data.length === 0) {
                savedConversationsList.innerHTML = '<div class="text-muted small">No saved conversations found.</div>';
                return;
            }
            
            // Add conversations to the list
            data.forEach(conversation => {
                const item = document.createElement('div');
                item.className = 'saved-conversation-item';
                item.dataset.file = conversation.file;
                
                const title = document.createElement('div');
                title.className = 'title';
                title.textContent = conversation.calculator || 'Conversation';
                
                const meta = document.createElement('div');
                meta.className = 'meta';
                
                // Format date from YYYYMMDD format
                const date = conversation.timestamp;
                const formattedDate = `${date.substring(0, 4)}-${date.substring(4, 6)}-${date.substring(6, 8)}`;
                
                meta.textContent = `${formattedDate} · ${conversation.message_count} messages`;
                
                item.appendChild(title);
                item.appendChild(meta);
                
                // Add click handler
                item.addEventListener('click', () => loadConversation(conversation.file));
                
                savedConversationsList.appendChild(item);
            });
        } catch (error) {
            console.error('Error loading saved conversations:', error);
            savedConversationsList.innerHTML = '<div class="text-danger small">Error loading conversations</div>';
        }
    }
    
    // Load a saved conversation
    async function loadConversation(fileName) {
        try {
            const response = await fetch('/api/load-conversation', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    file: fileName,
                    session_id: sessionId
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                // 1️⃣ Replace messages
                messagesContainer.innerHTML = '';
                if (Array.isArray(data.messages)) {
                    renderMessages(data.messages);
                }

                // 2️⃣ Restore calculator selection (if any)
                if (data.selected_calculator) {
                    selectedCalculator = data.selected_calculator;
                    selectedCalculatorUrl = data.calculator_url || '';

                    // Set dropdown value to match
                    for (let i = 0; i < calculatorSelect.options.length; i++) {
                        const opt = calculatorSelect.options[i];
                        if (opt.value === selectedCalculator) {
                            calculatorSelect.selectedIndex = i;
                            break;
                        }
                    }
                    updateCalculatorUI();
                }

                // 3️⃣ Restore model dropdown
                if (data.model) {
                    modelSelect.value = data.model;
                }

                showNotification('Conversation loaded', 'success');
            } else {
                showNotification(data.message, 'error');
            }
        } catch (error) {
            console.error('Error loading conversation:', error);
            showNotification('Failed to load conversation', 'error');
        }
    }
    
    // Setup event listeners
    function setupEventListeners() {
        // Send message when clicking send button
        sendButton.addEventListener('click', sendMessage);
        
        // Send message when pressing Enter (but not with Shift)
        userInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
        
        // Calculator selection change
        calculatorSelect.addEventListener('change', function() {
            const selectedOption = calculatorSelect.options[calculatorSelect.selectedIndex];
            if (selectedOption.value) {
                selectedCalculator = selectedOption.value;
                selectedCalculatorUrl = selectedOption.dataset.url;
                
                // Update UI
                calculatorUrl.textContent = selectedOption.value;
                calculatorUrl.href = selectedOption.dataset.url;
                calculatorUrlContainer.classList.remove('d-none');
                useCalculatorBtn.classList.remove('d-none');
            }
        });
        
        // Use calculator button
        useCalculatorBtn.addEventListener('click', function() {
            if (selectedCalculator && selectedCalculatorUrl) {
                if (websocket && websocket.readyState === WebSocket.OPEN) {
                    websocket.send(JSON.stringify({
                        type: 'select_calculator',
                        calculator_name: selectedCalculator,
                        calculator_url: selectedCalculatorUrl
                    }));
                } else {
                    // Fallback to REST API if WebSocket is not available
                    selectCalculator(selectedCalculator, selectedCalculatorUrl);
                }
            }
        });
        
        // Model selection change
        modelSelect.addEventListener('change', function() {
            if (websocket && websocket.readyState === WebSocket.OPEN) {
                websocket.send(JSON.stringify({
                    type: 'update_model',
                    model: modelSelect.value
                }));
            } else {
                // Fallback to REST API
                updateModel(modelSelect.value);
            }
        });
        
        // New conversation button
        newConversationBtn.addEventListener('click', function() {
            if (confirm('Start a new conversation? This will clear the current chat.')) {
                if (websocket && websocket.readyState === WebSocket.OPEN) {
                    websocket.send(JSON.stringify({
                        type: 'new_conversation'
                    }));
                } else {
                    // Fallback to REST API
                    startNewConversation();
                }
            }
        });
        
        // Save conversation button
        saveConversationBtn.addEventListener('click', function() {
            if (websocket && websocket.readyState === WebSocket.OPEN) {
                websocket.send(JSON.stringify({
                    type: 'save_conversation'
                }));
            } else {
                // Fallback to REST API
                saveConversation();
            }
        });
    }
    
    // Send a message
    function sendMessage() {
        const message = userInput.value.trim();
        
        if (!message) return;
        
        // Check if calculator is selected
        if (!selectedCalculator) {
            showNotification('Please select a calculator first', 'error');
            return;
        }
        
        // Add user message to chat
        addMessageToChat('user', message);
        
        // Clear input
        userInput.value = '';
        
        // Show typing indicator
        showTypingIndicator();
        
        // Send message via WebSocket
        if (websocket && websocket.readyState === WebSocket.OPEN) {
            websocket.send(JSON.stringify({
                type: 'chat',
                message: message
            }));
        } else {
            // Fallback to REST API if WebSocket is not available
            sendMessageViaRest(message);
        }
    }
    
    // Fallback: Send message via REST API
    async function sendMessageViaRest(message) {
        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    message: message,
                    session_id: sessionId
                })
            });
            
            const data = await response.json();
            
            // Remove typing indicator
            removeTypingIndicator();
            
            if (data.success) {
                // Add assistant response to chat
                addMessageToChat('assistant', data.response);
            } else {
                showNotification(data.response, 'error');
            }
        } catch (error) {
            console.error('Error sending message:', error);
            removeTypingIndicator();
            showNotification('Failed to send message', 'error');
        }
    }
    
    // Fallback: Select calculator via REST API
    async function selectCalculator(calculatorName, calculatorUrl) {
        try {
            const response = await fetch('/api/select-calculator', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    calculator_name: calculatorName,
                    calculator_url: calculatorUrl,
                    session_id: sessionId
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                selectedCalculator = calculatorName;
                selectedCalculatorUrl = calculatorUrl;
                addMessageToChat('assistant', data.message);
            } else {
                showNotification('Failed to select calculator', 'error');
            }
        } catch (error) {
            console.error('Error selecting calculator:', error);
            showNotification('Failed to select calculator', 'error');
        }
    }
    
    // Fallback: Update model via REST API
    async function updateModel(model) {
        try {
            const response = await fetch(`/api/session/${sessionId}/update-model`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    model: model
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                showNotification(`Model updated to ${model}`);
            } else {
                showNotification('Failed to update model', 'error');
            }
        } catch (error) {
            console.error('Error updating model:', error);
            showNotification('Failed to update model', 'error');
        }
    }
    
    // Fallback: Start new conversation via REST API
    async function startNewConversation() {
        try {
            const response = await fetch('/api/new-conversation', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    session_id: sessionId
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                // Reload the page to refresh UI
                window.location.reload();
            } else {
                showNotification('Failed to start new conversation', 'error');
            }
        } catch (error) {
            console.error('Error starting new conversation:', error);
            showNotification('Failed to start new conversation', 'error');
        }
    }
    
    // Fallback: Save conversation via REST API
    async function saveConversation() {
        try {
            const response = await fetch('/api/save-conversation', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    session_id: sessionId
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                showNotification('Conversation saved successfully!');
                // Reload saved conversations list
                loadSavedConversations();
            } else {
                showNotification('Failed to save conversation', 'error');
            }
        } catch (error) {
            console.error('Error saving conversation:', error);
            showNotification('Failed to save conversation', 'error');
        }
    }
    
    // Add a message to the chat
    function addMessageToChat(role, content) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}-message`;
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        // Process markdown in content
        const renderedContent = marked.parse(content);
        contentDiv.innerHTML = renderedContent;
        
        // Add time
        const timeDiv = document.createElement('div');
        timeDiv.className = 'message-time';
        timeDiv.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        
        messageDiv.appendChild(contentDiv);
        messageDiv.appendChild(timeDiv);
        
        messagesContainer.appendChild(messageDiv);
        
        // Scroll to bottom
        scrollToBottom();
    }
    
    // Show typing indicator
    function showTypingIndicator() {
        if (isTyping) return;
        
        isTyping = true;
        
        const typingDiv = document.createElement('div');
        typingDiv.className = 'typing-indicator';
        typingDiv.id = 'typingIndicator';
        
        for (let i = 0; i < 3; i++) {
            const dot = document.createElement('span');
            typingDiv.appendChild(dot);
        }
        
        messagesContainer.appendChild(typingDiv);
        scrollToBottom();
    }
    
    // Remove typing indicator
    function removeTypingIndicator() {
        const typingIndicator = document.getElementById('typingIndicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
        isTyping = false;
    }
    
    // Show notification
    function showNotification(message, type = 'success') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        
        // Add to body
        document.body.appendChild(notification);
        
        // Show animation
        setTimeout(() => {
            notification.classList.add('show');
        }, 10);
        
        // Hide after timeout
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => {
                notification.remove();
            }, 300);
        }, 3000);
    }
    
    // Update calculator UI
    function updateCalculatorUI() {
        if (selectedCalculator) {
            // Find the option matching the selected calculator
            for (let i = 0; i < calculatorSelect.options.length; i++) {
                if (calculatorSelect.options[i].value === selectedCalculator) {
                    calculatorSelect.selectedIndex = i;
                    break;
                }
            }
            
            // Update URL display
            calculatorUrl.textContent = selectedCalculator;
            calculatorUrl.href = selectedCalculatorUrl;
            calculatorUrlContainer.classList.remove('d-none');
            useCalculatorBtn.classList.remove('d-none');
        } else {
            // Reset selection
            calculatorSelect.selectedIndex = 0;
            calculatorUrlContainer.classList.add('d-none');
            useCalculatorBtn.classList.add('d-none');
        }
    }
    
    // Render messages from history
    function renderMessages(messages) {
        messagesContainer.innerHTML = '';
        
        if (!messages || messages.length === 0) {
            // Show welcome message if no messages
            const welcomeDiv = document.createElement('div');
            welcomeDiv.className = 'welcome-message';
            
            welcomeDiv.innerHTML = `
                <div class="welcome-icon">🏥</div>
                <div class="welcome-title">MedCalc-Agent</div>
                <div>👋 Hi, I'm MedCalc-Agent! I am a browser-augmented LLM agent who can help you with medical calculations and risk assessments! Please select a calculator from the sidebar to get started. 😊</div>
            `;
            
            messagesContainer.appendChild(welcomeDiv);
            return;
        }
        
        // Render all messages
        messages.forEach(message => {
            addMessageToChat(message.role, message.content);
        });
    }
    
    // Scroll to bottom of messages container
    function scrollToBottom() {
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
    
    // Generate a random session ID
    function generateSessionId() {
        return 'session_' + Math.random().toString(36).substring(2, 15);
    }
});

// Add notification styling
const style = document.createElement('style');
style.textContent = `
.notification {
    position: fixed;
    top: 20px;
    right: 20px;
    padding: 10px 20px;
    border-radius: 4px;
    color: white;
    font-weight: 500;
    opacity: 0;
    transform: translateY(-20px);
    transition: opacity 0.3s, transform 0.3s;
    z-index: 1000;
    max-width: 300px;
}

.notification.success {
    background-color: #4caf50;
}

.notification.error {
    background-color: #f44336;
}

.notification.show {
    opacity: 1;
    transform: translateY(0);
}
`;
document.head.appendChild(style); 