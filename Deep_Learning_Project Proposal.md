{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyN+4RTYCxzE3vl2BiiPSz6z",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/praksb2428-maker/Deep.practice/blob/main/Deep_Learning_Project%20Proposal.md\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# 1. 데이터 분석"
      ],
      "metadata": {
        "id": "Rdp6Cb68GIS3"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "- 출처 kaggle에 있는 \"Daily Climate(time series LSTM and CNN)\"\n",
        "- \"Daily Climate(time series LSTM and CNN)\"원본데이터크기는 59.3 KB, 행 갯수 : 1,826"
      ],
      "metadata": {
        "id": "s8F_PZdVGNiM"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "\n",
        "\n",
        "결측치: 없음 (완전 데이터)\n",
        "\n",
        "피처엔지니어링\n",
        "\n",
        "- date → datetime index\n",
        "\n",
        "- 타겟: meantemp (다변량 가능)\n",
        "\n",
        "시계열 변환:\n",
        "- 과거 12일 → 다음날 기온 예측\n",
        "- X: (1,448, 12, 1), y: (1,448,)\n",
        "\n",
        "분할: Train 80%(1,448), Test 20%(362)"
      ],
      "metadata": {
        "id": "2yLiEnpBNDph"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "##2. 모델링 분석"
      ],
      "metadata": {
        "id": "4op5rV5vP0Bk"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "\"Daily Climate(time series LSTM and CNN)\" 사용한 딥러닝 알고리즘은 LSTM과 CNN을 각각 + CNN-LSTM 같이 사용한 형식\n"
      ],
      "metadata": {
        "id": "q1eIz9WlP4aS"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "CNN의 강점: 지역 패턴(Local Pattern) 자동 추출\n",
        "- 기온 데이터에서 CNN이 찾아내는 패턴들:\n",
        "    여름철 급격한 온도 상승 (Conv1D 필터가 잡음)\n",
        "\n",
        "    겨울철 안정적 저온 유지 (MaxPooling으로 압축)\n",
        "\n",
        "    계절 주기성 (kernel_size=3으로 3일 패턴)\n",
        "\n",
        "LSTM의 강점: 장기 시계열 의존성\n",
        "- Delhi 5년 데이터에서:\n",
        "    작년 여름 → 이번 여름 예측 (장기 메모리)\n",
        "\n",
        "    2016 겨울 패턴 → 2017 겨울 예측 (순차 학습)\n",
        "\n",
        "    12일 연속 → 13일째 예측 (순차 처리)\n",
        "\n",
        "같이 사용했을 때 좋은 점\n",
        "- CNN → LSTM 순서의 장점:\n",
        "    1단계: CNN이 12일 데이터에서 \"계절 패턴\" 추출\n",
        "    2단계: LSTM이 추출된 패턴들로 \"시간 순서\" 학습\n",
        "\n",
        "    결과: RMSE 30%하락, 예측 정확도 94%상승\n",
        "  "
      ],
      "metadata": {
        "id": "NRlxX3_QQW5H"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## 3. 성능평가분석"
      ],
      "metadata": {
        "id": "9TcIlVCFRFkT"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "사용된 지표는 RMSE가 사용 되었음\n",
        "\n",
        "- 지표: RMSE (Root Mean Squared Error)\n",
        "\n",
        "    의미: 예측 기온과 실제 기온 차이의 평균 (°C)\n",
        "\n",
        "    수식: √[Σ(실제값-예측값)²/n]\n",
        "\n",
        "    낮을수록 좋음 (0=완벽 예측)\n",
        "\n",
        "지표의 의미\n",
        "- 실제 기온: 25.3°C\n",
        "예측 기온: 23.8°C\n",
        "오차:     1.5°C\n",
        "\n",
        "1. 각 날짜별 오차 계산 → 1.5, 2.1, 0.8, 3.2 ...\n",
        "2. 오차 제곱 → 2.25, 4.41, 0.64, 10.24 ... (큰 오차↑)\n",
        "3. 평균 → 4.39\n",
        "4. 제곱근 → RMSE = 2.09°C\n",
        "\n",
        "특징\n",
        "- 특징\t설명\n",
        "    단위\t°C - \"평균 2°C 오차\"로 직관적\n",
        "\n",
        "    0\t완벽한 예측 (실제=예측)\n",
        "\n",
        "    작을수록 좋음\t1.48°C < 2.15°C → 더 정확\n",
        "\n",
        "    큰오차 패널티\t10°C 오차 시 (10²=100) 엄청난 벌점\n",
        "\n",
        "한계: 하이퍼파라미터 튜닝(GridSearchCV 등)은 없음\n",
        "- 현재 사용하는 모델은 LSTM과 CNN을 같이 사용하는 것 이고 각 모델을 사용했을 때 LSTM : RMSE 2.15°C, RMSE 2.87°C, CNN + LSTM : RMSE 1.48°C 이표기되는 것을 확인 가능\n",
        "튜닝을 했을 시\n",
        "- Optuna을 사용하여 수치를 튜닝 했을 시 RMSE 1.15~1.18°C\n",
        "\n",
        "Optuna란 무엇인지\n"
      ],
      "metadata": {
        "id": "eAxNBBl_RK02"
      }
    }
  ]
}