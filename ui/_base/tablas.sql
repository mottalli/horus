CREATE TABLE base_iris (
        id_imagen INTEGER PRIMARY KEY AUTOINCREMENT,
	nombre TEXT NOT NULL,
        imagen TEXT NOT NULL,
        segmentacion TEXT,
        codigo_gabor TEXT
);

CREATE TABLE comparaciones (
        id_imagen1 INTEGER NOT NULL,
        id_imagen2 INTEGER NOT NULL,
        distancia FLOAT NOT NULL,
        intra_clase INTEGER NOT NULL,
        PRIMARY KEY(id_imagen1, id_imagen2)
);


CREATE TABLE comparaciones_a_contrario (
        id_imagen1 INTEGER NOT NULL,
        id_imagen2 INTEGER NOT NULL,
        distancia FLOAT NOT NULL,
        parte INTEGER NOT NULL,
        intra_clase INTEGER NOT NULL,
        PRIMARY KEY(id_imagen1, id_imagen2, parte)
);

CREATE INDEX caa_idx ON comparaciones_a_contrario(id_imagen1);

CREATE TABLE nfa_a_contrario (
        id_imagen1 INTEGER NOT NULL,
        id_imagen2 INTEGER NOT NULL,
        nfa FLOAT NOT NULL,
        intra_clase INTEGER NOT NULL,
        PRIMARY KEY(id_imagen1, id_imagen2)
);

