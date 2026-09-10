# SwarVed AI — System Architecture

```mermaid
flowchart TB

    %% =========================
    %% MOBILE DEVICE LAYER
    %% =========================

    subgraph MOBILE["📱 MOBILE DEVICE LAYER"]

        A["📞 Incoming Call"]

        B["🎧 100ms RAM Audio Buffer"]

        C["🔊 C++ RNNoise Filter"]

        D["🧠 C++ ONNX Conformer Engine<br/><45ms Detection"]

        E["🚨 SYSTEM_ALERT_WINDOW<br/>Red Warning Overlay"]

        A --> B
        B --> C
        C --> D
        D --> E

    end


    %% =========================
    %% ENFORCEMENT LAYER
    %% =========================

    subgraph ENFORCEMENT["🛡️ ENFORCEMENT LAYER"]

        F["⚡ 1-Tap Emergency Protection"]

        G["💳 UPI Safety / Freeze Workflow"]

        H["GPay / PhonePe / Banking Security"]

        F --> G
        G --> H

    end


    %% =========================
    %% GOVERNMENT LAYER
    %% =========================

    subgraph GOVERNMENT["🏛️ GOVERNMENT DPI GATEWAY"]

        I["⚙️ FastAPI Microservice"]

        J["📞 Cybercrime Reporting Workflow<br/>1930 / Authorized Portal"]

        I --> J

    end


    %% =========================
    %% CONNECTIONS
    %% =========================

    E --> F

    F --> I


    %% =========================
    %% COLORS
    %% =========================

    classDef mobile fill:#0F172A,color:#FFFFFF,stroke:#22D3EE,stroke-width:2px
    classDef danger fill:#3F1115,color:#FFFFFF,stroke:#EF4444,stroke-width:3px
    classDef protection fill:#132B25,color:#FFFFFF,stroke:#22C55E,stroke-width:2px
    classDef government fill:#1E293B,color:#FFFFFF,stroke:#A78BFA,stroke-width:2px

    class A,B,C,D mobile
    class E danger
    class F,G,H protection
    class I,J government
