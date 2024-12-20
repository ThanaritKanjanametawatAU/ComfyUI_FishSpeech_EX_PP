function createGradioAnimation() {
    const params = new URLSearchParams(window.location.search);
    if (!params.has('__theme')) {
        params.set('__theme', 'dark');
        window.location.search = params.toString();
    }

    var gradioApp = document.querySelector('gradio-app');
    if (gradioApp) {
        document.documentElement.style.setProperty('--my-200', '#4ade80');
        document.documentElement.style.setProperty('--my-50', '#1a1a1a');
        document.documentElement.style.setProperty('--primary-color', '#4ade80');
        document.documentElement.style.setProperty('--secondary-color', '#22c55e');
        document.documentElement.style.setProperty('--accent-color', '#86efac');
        document.documentElement.style.setProperty('--bg-dark', '#1a1a1a');
        document.documentElement.style.setProperty('--text-dark', '#e0e0e0');
    }

    var container = document.createElement('div');
    container.id = 'gradio-animation';
    container.style.fontSize = '2em';
    container.style.fontFamily = 'Maiandra GD, ui-monospace, monospace';
    container.style.fontWeight = 'bold';
    container.style.textAlign = 'center';
    container.style.marginBottom = '20px';
    container.style.color = '#4ade80';

    var text = 'Welcome to Fish-Speech!';
    for (var i = 0; i < text.length; i++) {
        (function(i){
            setTimeout(function(){
                var letter = document.createElement('span');
                letter.style.opacity = '0';
                letter.style.transition = 'opacity 0.5s';
                letter.innerText = text[i];

                container.appendChild(letter);

                setTimeout(function() {
                    letter.style.opacity = '1';
                }, 50);
            }, i * 200);
        })(i);
    }

    var gradioContainer = document.querySelector('.gradio-container');
    gradioContainer.insertBefore(container, gradioContainer.firstChild);

    return 'Animation created';
}
