const API_URL = '/api';

// Initialize chat UI
document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chatForm');
    const userInput = document.getElementById('userInput');
    const chatContainer = document.getElementById('chatContainer');

    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const message = userInput.value.trim();
        if (!message) return;

        // Add user message to chat
        appendMessage('user', message);
        userInput.value = '';

        try {
            // Send message to API
            const response = await fetch(`${API_URL}/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-Key': process.env.API_KEY
                },
                body: JSON.stringify({ message })
            });

            if (!response.ok) {
                throw new Error('API request failed');
            }

            const data = await response.json();
            appendMessage('assistant', data.response);
        } catch (error) {
            console.error('Error:', error);
            appendMessage('system', 'Sorry, something went wrong. Please try again.');
        }
    });

    function appendMessage(role, content) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role} mb-4`;
        
        const colors = {
            user: 'bg-blue-100 dark:bg-blue-900',
            assistant: 'bg-gray-100 dark:bg-gray-800',
            system: 'bg-red-100 dark:bg-red-900'
        };

        messageDiv.innerHTML = `
            <div class="p-4 rounded-lg ${colors[role]}">
                <p class="text-sm text-gray-600 dark:text-gray-400">${role}</p>
                <p class="mt-1">${content}</p>
            </div>
        `;

        chatContainer.appendChild(messageDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
});