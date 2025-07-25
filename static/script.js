document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prompt-form');
    const promptInput = document.getElementById('prompt-input');
    const submitButton = document.getElementById('submit-button');
    const loadingSpinner = document.getElementById('loading-spinner');
    const chunksContainer = document.getElementById('relevant-chunks-container');
    const chunksOutput = document.getElementById('relevant-chunks-output');
    const responseOutput = document.getElementById('response-output');

    form.addEventListener('submit', async (event) => {
        event.preventDefault(); // Prevent default form submission

        const prompt = promptInput.value.trim();
        if (!prompt) {
            alert('Please enter a prompt.');
            return;
        }

        // --- Reset UI and show loading state ---
        submitButton.disabled = true;
        loadingSpinner.classList.remove('hidden');
        chunksContainer.classList.add('hidden'); // Hide previous chunks
        chunksOutput.innerHTML = ''; // Clear previous chunks
        responseOutput.textContent = ''; // Clear previous response

        try {
            // --- NEW: Single API call to the '/chat' endpoint ---
            console.log("Sending prompt to the /chat endpoint...");

            // <-- CHANGE: We now make only ONE fetch request to the new /chat endpoint
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                // <-- CHANGE: We only need to send the prompt now
                body: JSON.stringify({ prompt: prompt })
            });

            if (!response.ok) {
                // Try to parse error, otherwise use status text
                let errorMsg = response.statusText;
                try {
                    const errorData = await response.json();
                    errorMsg = errorData.error || errorMsg;
                } catch (e) { /* Ignore if response is not JSON */ }
                throw new Error(`Failed to get response: ${errorMsg}`);
            }

            // <-- CHANGE: The single response contains both chunks and the LLM response
            const data = await response.json();
            const relevantChunks = data.relevant_chunks;
            const llmResponse = data.llm_response;

            console.log("Success: Received combined data:", data);

            // --- Display the relevant chunks in the UI ---
            if (relevantChunks && relevantChunks.length > 0) {
                relevantChunks.forEach(chunk => {
                    const chunkElement = document.createElement('div');
                    chunkElement.className = 'chunk';
                    // Using innerHTML is fine here as the content is controlled
                    chunkElement.innerHTML = `<p><strong>ID: ${chunk.id}</strong> (Distance: ${chunk.distance.toFixed(4)})</p><pre>${chunk.text}</pre>`;
                    chunksOutput.appendChild(chunkElement);
                });
                chunksContainer.classList.remove('hidden'); // Show the container
            } else {
                 chunksOutput.innerHTML = '<p>No relevant chunks found from your documents.</p>';
                 chunksContainer.classList.remove('hidden');
            }

            // --- Display the final LLM response ---
            responseOutput.textContent = llmResponse;

        } catch (error) {
            console.error(error);
            responseOutput.textContent = `Error: ${error.message}`;
        } finally {
            // --- Hide loading indicator and re-enable the button ---
            submitButton.disabled = false;
            loadingSpinner.classList.add('hidden');
        }
    });
});