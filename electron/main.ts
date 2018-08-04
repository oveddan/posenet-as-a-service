import {app, BrowserView, BrowserWindow} from 'electron'

let win;

function createWindow() {
  win = new BrowserWindow({width: 800, height: 800});
  win.on('closed', () => {
    win = null;
  });

  win.loadURL('http://localhost:1234');
  // let view = new BrowserView({webPreferences: {nodeIntegration: false}});

  // win.setBrowserView(view);

  // view.setBounds({x: 0, y: 0, width: 800, height: 800});

  // view.webContents.loadURL('http://localhost:1234');
}

app.on('ready', createWindow);
