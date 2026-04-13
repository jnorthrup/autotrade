mp-superproject* - just a convenience git rep
-columnar*
-mp*
--acapulco
--control
-hrm*
-binance*

 * = root with no parent pom

otherwise parent relationships appear as above.
mp-superproject (parent pom)
├── hrm
│   └── pom.xml
├── columnar
│   └── pom.xml
└── mp
|    ├── acapulco
|    │   └── pom.xml
|    └── control
|        └── pom.xml
+---binance
