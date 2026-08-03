document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const form = document.getElementById('convertForm');
    const statusMessage = document.getElementById('statusMessage');
    const submitBtn = document.getElementById('submitBtn');
    const btnLoader = document.getElementById('btnLoader');
    const btnText = document.getElementById('btnText');
    const fileInfo = document.getElementById('fileInfo');
    const fileName = document.getElementById('fileName');
    const removeFile = document.getElementById('removeFile');

    let currentFile = null;

    // UI Interactions
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults (e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
    });

    dropZone.addEventListener('drop', handleDrop, false);
    dropZone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', (e) => handleFiles(e.target.files));
    removeFile.addEventListener('click', clearFile);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }

    function handleFiles(files) {
        if (files.length === 0) return;
        const file = files[0];
        
        if (!file.name.toLowerCase().endsWith('.docx')) {
            showStatus('Veuillez sélectionner un fichier .docx valide.', 'error');
            clearFile();
            return;
        }

        currentFile = file;
        fileName.textContent = file.name;
        dropZone.style.display = 'none';
        fileInfo.style.display = 'flex';
        submitBtn.classList.add('active');
        showStatus('', '');
    }

    function clearFile(e) {
        if (e) e.stopPropagation();
        currentFile = null;
        fileInput.value = '';
        dropZone.style.display = 'block';
        fileInfo.style.display = 'none';
        submitBtn.classList.remove('active');
        showStatus('', '');
    }

    function showStatus(text, type) {
        statusMessage.textContent = text;
        statusMessage.className = 'status-message show ' + type;
        
        if (!text) {
            statusMessage.className = 'status-message';
        }
    }

    function setLoading(isLoading) {
        if (isLoading) {
            btnLoader.style.display = 'inline-block';
            btnText.textContent = 'Conversion en cours...';
            submitBtn.style.pointerEvents = 'none';
            removeFile.style.pointerEvents = 'none';
        } else {
            btnLoader.style.display = 'none';
            btnText.textContent = 'Convertir en PDF';
            submitBtn.style.pointerEvents = 'auto';
            removeFile.style.pointerEvents = 'auto';
        }
    }

    // Form Submission
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (!currentFile) return;

        setLoading(true);
        showStatus('Génération de votre PDF haute fidélité...', 'loading');

        const formData = new FormData();
        formData.append('file', currentFile);

        try {
            const response = await fetch('/api/v1/document/convert/', {
                method: 'POST',
                body: formData,
            });

            if (response.ok) {
                showStatus('Conversion réussie ! Téléchargement en cours...', 'success');
                
                // Télécharger le Blob
                const blob = await response.blob();
                const downloadUrl = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.style.display = 'none';
                a.href = downloadUrl;
                
                // On récupère le nom original sans l'extension
                const baseName = currentFile.name.replace(/\.[^/.]+$/, "");
                a.download = `${baseName}.pdf`;
                
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(downloadUrl);
                
                // Réinitialisation après succès
                setTimeout(clearFile, 3000);
            } else {
                const errorData = await response.json();
                showStatus(errorData.error || 'Erreur lors de la conversion.', 'error');
            }
        } catch (error) {
            showStatus('Erreur de connexion au serveur.', 'error');
            console.error(error);
        } finally {
            setLoading(false);
        }
    });
});
