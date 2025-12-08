const { app, BrowserWindow } = require('electron');
function createWindow() {
    const win = new BrowserWindow({
        width: 1200,
        height: 900,
        webPreferences: { preload: __dirname + "/preload.js" }
    });

    // 若你部署在 Render
    win.loadURL("https://stellar-spectrum-web-2.onrender.com");

    // 若你想加载本地 Flask，请改为：
    // win.loadURL("http://127.0.0.1:5000");
}

app.whenReady().then(() => {
    createWindow();
});
