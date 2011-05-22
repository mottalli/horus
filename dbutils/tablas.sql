CREATE TABLE usuarios (
	id_usuario INTEGER PRIMARY KEY AUTOINCREMENT,
	nombre TEXT NOT NULL
);

CREATE TABLE base_iris (
	id_imagen INTEGER PRIMARY KEY AUTOINCREMENT,
	id_usuario INTEGER NOT NULL,
	imagen_primaria INTEGER NOT NULL DEFAULT 0,
	imagen TEXT NOT NULL,
	segmentacion TEXT NOT NULL,
	segmentacion_correcta INTEGER NOT NULL DEFAULT 1,
	iris_template TEXT NOT NULL,
	average_template TEXT
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

