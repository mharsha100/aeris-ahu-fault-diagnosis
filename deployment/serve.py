from pathlib import Path
import sys,os,subprocess,signal
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from aeris.service import start
server,service=start(ROOT,os.environ.get('AERIS_BIND','127.0.0.1'),8001)
child=subprocess.Popen([sys.executable,'-m','streamlit','run',str(ROOT/'app.py'),'--server.address='+os.environ.get('AERIS_BIND','127.0.0.1'),'--server.port=8501','--server.headless=true','--browser.gatherUsageStats=false'])
def stop(*args):
    child.terminate();server.shutdown()
    try:child.wait(timeout=10)
    except subprocess.TimeoutExpired:child.kill()
    raise SystemExit(0)
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
print('AERIS API/metrics:8001 UI:8501; selected bundle is validated lazily.',flush=True)
raise SystemExit(child.wait())
