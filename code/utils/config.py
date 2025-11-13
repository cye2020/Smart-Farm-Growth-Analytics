from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.parent
CODE_DIR = PROJECT_DIR / 'code'
DATA_DIR = PROJECT_DIR / 'data'
MODEL_DIR = PROJECT_DIR / 'models'
REPORT_DIR = PROJECT_DIR / 'reports'
<<<<<<< HEAD
FONT_DIR = PROJECT_DIR / 'font'
=======
>>>>>>> origin/yun

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

energy_map = {
    "measDate": "측정일",
    "farm_cde": "온실번호",
    "water_usage": "물사용량(L)",
    "water_cost": "물사용비용(원)",
    "fertilizer_usage": "비료사용량(L)",
    "fertilizer_cost": "비료사용비용(원)",
    "heating_energy_usage": "난방에너지사용량(kcal)",
    "heating_energy_cost": "난방에너지사용비용(원)",
    "CO2_usage": "CO₂사용량(L)",
    "CO2_cost": "CO₂사용비용(원)",
    "mist_usage_time": "미스트사용시간(분)",
    "mist_cost": "미스트사용비용(원)"
}
