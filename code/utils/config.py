from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.parent
CODE_DIR = PROJECT_DIR / 'code'
DATA_DIR = PROJECT_DIR / 'data'
MODEL_DIR = PROJECT_DIR / 'models'


growth_map = {
    "measDate": "측정일",
    "farm_cde": "온실번호",
    "itemCode": "품목코드",
    "flowerTop": "화방높이(cm)",
    "grwtLt": "생장길이(cm)",
    "lefCunt": "엽수(개)",
    "lefLt": "엽장(cm)",
    "lefBt": "엽폭(cm)",
    "stemThck": "줄기직경(mm)",
    "flanGrupp": "개화수준(점)",
    "frtstGrupp": "착과수준(점)",
    "hvstGrupp": "수확중 화방수준(점)",
    "frtstCo": "열매수(개)"
}
