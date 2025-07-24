document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prompt-form');
    const promptInput = document.getElementById('prompt-input');
    const submitButton = document.getElementById('submit-button');
    const loadingSpinner = document.getElementById('loading-spinner');

    // Get the new elements for displaying chunks
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
            // --- STEP 1: Fetch the relevant chunks from the backend ---
            console.log("Step 1: Fetching relevant chunks...");
            const chunksResponse = await fetch('/get_relevant_chunks', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prompt: prompt })
            });

            if (!chunksResponse.ok) {
                // Try to parse error, otherwise use status text
                let errorMsg = chunksResponse.statusText;
                try {
                    const errorData = await chunksResponse.json();
                    errorMsg = errorData.error || errorMsg;
                } catch (e) { /* Ignore if response is not JSON */ }
                throw new Error(`Failed to fetch chunks: ${errorMsg}`);
            }

            const chunksData = await chunksResponse.json();
            const relevantChunks = chunksData.relevant_chunks;
            console.log("Step 1 Success: Received chunks:", relevantChunks);


            // --- STEP 2: Display the relevant chunks in the UI ---
            if (relevantChunks && relevantChunks.length > 0) {
                relevantChunks.forEach(chunk => {
                    const chunkElement = document.createElement('div');
                    chunkElement.className = 'chunk';
                    chunkElement.innerHTML = `<p><strong>ID: ${chunk.id}</strong> (Distance: ${chunk.distance.toFixed(4)})</p><pre>${chunk.text}</pre>`;
                    chunksOutput.appendChild(chunkElement);
                });
                chunksContainer.classList.remove('hidden'); // Show the container
            } else {
                 chunksOutput.innerHTML = '<p>No relevant chunks found.</p>';
            }


            // --- STEP 3: Pass the prompt AND the chunks to the LLM to generate the final response ---
            //    (This assumes you have a '/generate' endpoint ready to accept this data)
            console.log("Step 3: Sending prompt and chunks to LLM for final response...");

            // NOTE: We are creating a hypothetical '/generate' endpoint call here.
            // You will need to create this endpoint in your Flask app.
            const llmResponse = await fetch('/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    prompt: prompt,
                    // Send the text of the chunks to the LLM
                    chunks: relevantChunks.map(c => c.text)
                })
            });

            if (!llmResponse.ok) {
                let errorMsg = llmResponse.statusText;
                try {
                    const errorData = await llmResponse.json();
                    errorMsg = errorData.error || errorMsg;
                } catch (e) { /* Ignore if response is not JSON */ }
                throw new Error(`LLM generation failed: ${errorMsg}`);
            }

            const finalData = await llmResponse.json();
            responseOutput.textContent = finalData.response;
            console.log("Step 3 Success: Received final response.");


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