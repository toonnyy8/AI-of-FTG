let keyboardJS = require("keyboardjs")
const ipcRenderer = require('electron').ipcRenderer;
keyboardJS.bind('f11', function () {
    ipcRenderer.send('full-screen');
});
keyboardJS.bind('f12', function () {
    ipcRenderer.send('DevTools');
});