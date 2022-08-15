# -*- mode: python ; coding: utf-8 -*-
import os

from PyInstaller.building.datastruct import TOC
from PyInstaller.utils.hooks import collect_all

CWD = os.getcwd()


datas = [('./models', './models'), ('./icon.ico', '.')]
binaries = []
hiddenimports = []
tmp_ret = collect_all('spacy_udpipe')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('sklearn.feature_extraction.text')

datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('sklearn.ensemble')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


block_cipher = None


a = Analysis(['app.py'],
             pathex=[],
             binaries=binaries,
             datas=datas,
             hiddenimports=hiddenimports,
             hookspath=[],
             hooksconfig={},
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)

# error - file already exists but it shouldn't
xs = [
    'cp36-win_amd64',
    'cp37-win_amd64',
]
datas_upd = TOC()

for d in a.datas:
    if not any([x in d[0] and x in d[1] for x in xs]):
        datas_upd.append(d)

a.datas = datas_upd


pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,  
          [],
          name='kaz_clf',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=False,
          disable_windowed_traceback=False,
          target_arch=None,
          codesign_identity=None,
          entitlements_file=None,
          icon=f'{CWD}/icon.ico')
