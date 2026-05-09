function updateClock() {

    const now = new Date();

    document.getElementById('clock').innerText =
        now.toLocaleTimeString();
}

setInterval(updateClock, 1000);

updateClock();

async function updateStatus() {

    try {

        const response = await fetch('/status');

        const data = await response.json();

        const statusEl =
            document.getElementById('system-status');

        const defectText =
            document.getElementById('defect-text');

        const fps =
            document.getElementById('fps');

        const confText =
            document.getElementById('confidence-text');

        const confFill =
            document.getElementById('confidence-fill');

        const alertBox =
            document.getElementById('alert-box');

        const alertMessage =
            document.getElementById('alert-message');

        fps.innerText = data.fps;

        const confPercent =
            Math.round(data.confidence * 100);

        confText.innerText = confPercent + '%';

        confFill.style.width =
            confPercent + '%';

        if (data.status === 'OK') {

            statusEl.innerText = 'OK';

            statusEl.className =
                'big-status ok';

            defectText.innerText =
                'No defects detected';

            alertBox.style.display = 'none';

        }

        else {

            statusEl.innerText = 'DEFECT';

            statusEl.className =
                'big-status defect';

            const defects =
                [...new Set(data.defects)];

            defectText.innerText =
                defects.join(', ');

            alertBox.style.display = 'block';

            alertMessage.innerText =
                'Detected: ' + defects.join(', ');
        }

    }

    catch (e) {
        console.log(e);
    }
}

setInterval(updateStatus, 200);

updateStatus();