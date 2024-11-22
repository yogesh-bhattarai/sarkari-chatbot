async function sendQuery() {
    const userInput = document.getElementById('user-input').value;
    if (!userInput) return alert("Please enter a question!");

    const chatBox = document.getElementById('chat-box');
    chatBox.innerHTML += `<p><strong>You:</strong> ${userInput}</p>`;

    try {
        const response = await fetch('/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: userInput }),
        });

        const data = await response.json();
        if (data.response) {
            chatBox.innerHTML += `<p><strong>Bot:</strong> ${data.response}</p>`;
        } else {
            chatBox.innerHTML += `<p><strong>Bot:</strong> Error: ${data.error}</p>`;
        }
    } catch (error) {
        chatBox.innerHTML += `<p><strong>Bot:</strong> Something went wrong!</p>`;
    }

    document.getElementById('user-input').value = '';
    chatBox.scrollTop = chatBox.scrollHeight;
}
