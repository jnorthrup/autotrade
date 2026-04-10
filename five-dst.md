# Five-DST Structure Overview

```mmd

chart TD
    subgraph "Phase 1: Trust Formation"
        direction LR
        F1[Fiduciary 1] -->|Appointed as Trustee| T1[Irrevocable Spendthrift Trust A]
        F2[Fiduciary 2] -->|Appointed as Trustee| T2[Irrevocable Spendthrift Trust B]
        F3[Fiduciary 3] -->|Appointed as Trustee| T3[Irrevocable Spendthrift Trust C]
        F4[Fiduciary 4] -->|Appointed as Trustee| T4[Irrevocable Spendthrift Trust D]
        F5[Fiduciary 5] -->|Appointed as Trustee| T5[Irrevocable Spendthrift Trust E]
        
        F1 -->|Cross-Trustee| T2
        F1 -->|Cross-Trustee| T3
        F1 -->|Cross-Trustee| T4
        F1 -->|Cross-Trustee| T5
        
        F2 -->|Cross-Trustee| T1
        F2 -->|Cross-Trustee| T3
        F2 -->|Cross-Trustee| T4
        F2 -->|Cross-Trustee| T5
        
        F3 -->|Cross-Trustee| T1
        F3 -->|Cross-Trustee| T2
        F3 -->|Cross-Trustee| T4
        F3 -->|Cross-Trustee| T5
        
        F4 -->|Cross-Trustee| T1
        F4 -->|Cross-Trustee| T2
        F4 -->|Cross-Trustee| T3
        F4 -->|Cross-Trustee| T5
        
        F5 -->|Cross-Trustee| T1
        F5 -->|Cross-Trustee| T2
        F5 -->|Cross-Trustee| T3
        F5 -->|Cross-Trustee| T4
        
        TP1[Trust Protector A] -->|Oversight| F2 & F3 & F4 & F5
        TP2[Trust Protector B] -->|Oversight| F1 & F3 & F4 & F5
        TP3[Trust Protector C] -->|Oversight| F1 & F2 & F4 & F5
        TP4[Trust Protector D] -->|Oversight| F1 & F2 & F3 & F5
        TP5[Trust Protector E] -->|Oversight| F1 & F2 & F3 & F4
        
        T1 -->|Beneficiary Conduit| DYN1[Dynasty Trust 1]
        T2 -->|Beneficiary Conduit| DYN2[Dynasty Trust 2]
        T3 -->|Beneficiary Conduit| DYN3[Dynasty Trust 3]
        T4 -->|Beneficiary Conduit| DYN4[Dynasty Trust 4]
        T5 -->|Beneficiary Conduit| DYN5[Dynasty Trust 5]
    end
    
    subgraph "Phase 2: D-ST Incorporation Focal Point"
        direction TB
        INT1[Intangible Asset 1<br/>Digital Assets]
        INT2[Intangible Asset 2<br/>Contractual Rights]
        INT3[Intangible Asset 3<br/>Intellectual Property]
        INT4[Intangible Asset 4<br/>Securitized Receivables]
        INT5[Intangible Asset 5<br/>Insurance Wrappers]
        
        T1 -->|Contributes| INT1
        T2 -->|Contributes| INT2
        T3 -->|Contributes| INT3
        T4 -->|Contributes| INT4
        T5 -->|Contributes| INT5
        
        INT1 & INT2 & INT3 & INT4 & INT5 -->|Pools Assets| DST[Delaware Statutory Trust<br/>Orphan SPV Structure]
        
        OT[Orphan Trust<br/>Charitable/STAR] -->|Holds Ownership| DST
        IT[Independent Institutional Trustee] -->|Manages| DST
        IT -->|Bankruptcy Filing Veto| DST
        
        DST -->|Issues| DEB[D-ST Debenture Instrument<br/>Bankruptcy-Remote Capacity Shares]
    end
    
    subgraph "Phase 3: Securities & Banking Infrastructure"
        direction LR
        DEB -->|File via API| EDGAR[SEC EDGAR System<br/>Form D Private Placement]
        
        DST -->|Establishes| ODFI[ACH ODFI Relationship<br/>Originator Depository Financial Institution]
        ODFI -->|Generates| ACH[ACH Files<br/>Settlement & Distribution]
        
        API[RESTful API Gateway] -->|Manages| DST
        API -->|Routes| ACH
        API -->|Tracks| LEDGER[Immutable Ledger<br/>Asset Performance]
    end
    
    subgraph "Phase 4: Card Issuance & Credit Facility"
        direction TB
        DEB -->|Collateralizes| COLL[Segregated Collateral Account<br/>110% Over-collateralized]
        COLL -->|Secures| CARD[ATM/Credit Card Program<br/>Bankruptcy-Remote Backing]
        
        BANK[Partner Bank/Issuer] -->|Issues| CARD
        ACH -->|Settles Transactions| CARD
        
        CARD -->|Provides| HOLDER[Card Holder<br/>Full Credit on Account]
        
        subgraph "Legal Firewall"
            RECOURSE[No Recourse to Legacy Trusts<br/>No Recourse to Beneficiaries<br/>DST Assets Only]
        end
        
        CARD -.->|Protected by| RECOURSE
    end
    
    subgraph "Compliance & Risk Layer"
        direction LR
        OP1[Legal Opinion<br/>Bankruptcy-Remote Validity]
        OP2[Legal Opinion<br/>Securities Law Compliance]
        OP3[Legal Opinion<br/>Tax Treatment]
        
        INS[Fiduciary Insurance<br/>Cross-Trustee Coverage]
        
        SC[Substantive Consolidation Risk<br/>⚠️ Strict Separateness Required]
        
        OP1 & OP2 & OP3 -->|Required| DST
        INS -->|Covers| F1 & F2 & F3 & F4 & F5
        SC -.->|Threatens| T1 & T2 & T3 & T4 & T5
    end
    
    subgraph "Tax & Regulatory"
        direction TB
        TAX1[Grantor Trust Status<br/>Intentionally Defective]
        TAX2[DST Taxation<br/>Partnership Election Form 1065]
        TAX3[Banking Regulations<br/>Reg E & TILA Compliance]
        
        TAX1 -->|Applies to| T1 & T2 & T3 & T4 & T5
        TAX2 -->|Applies to| DST
        TAX3 -->|Applies to| CARD
    end
    
    style DST fill:#1a4480,stroke:#ff6b35,stroke-width:3px
    style CARD fill:#ff6b35,stroke:#000,stroke-width:2px
    style DEB fill:#ffd700,stroke:#000,stroke-width:2px
    style RECOURSE fill:#d32f2f,stroke:#000,stroke-width:2px,color:#fff
    style SC fill:#d32f2f,stroke:#000,stroke-width:2px,color:#fff

``` 