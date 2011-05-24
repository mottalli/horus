CREATE TABLE usuarios (
	id_usuario INTEGER PRIMARY KEY AUTOINCREMENT,
	nombre TEXT NOT NULL
);

CREATE TABLE base_iris (
	id_iris INTEGER PRIMARY KEY AUTOINCREMENT,
	id_usuario INTEGER NOT NULL,
	imagen TEXT NOT NULL,
	segmentacion TEXT NOT NULL,
	entrada_valida INTEGER NOT NULL DEFAULT 1,
	image_template TEXT,
	average_template TEXT			-- Opcional, template que se generó en una captura en ráfaga (tiene precedencia sobre el template asociado a la imagen)
);

CREATE VIEW vw_base_iris AS
	SELECT id_iris,usuarios.id_usuario,nombre,imagen,segmentacion,entrada_valida,COALESCE(average_template,image_template) AS template
	FROM base_iris NATURAL JOIN usuarios;

-- Tablas para correr los scripts de comparación

DROP TABLE comparaciones;
CREATE TABLE comparaciones (
	id_iris1 INTEGER NOT NULL,
	id_iris2 INTEGER NOT NULL,
	distancia FLOAT NOT NULL,
	intra_clase INTEGER NOT NULL,
	PRIMARY KEY(id_iris1, id_iris2)
);

DROP TABLE comparaciones_a_contrario;
CREATE TABLE comparaciones_a_contrario (
	id_iris1 INTEGER NOT NULL,
	id_iris2 INTEGER NOT NULL,
	distancia FLOAT NOT NULL,
	parte INTEGER NOT NULL,
	intra_clase INTEGER NOT NULL,
	PRIMARY KEY(id_iris1, id_iris2, parte)
);

CREATE INDEX caa_idx ON comparaciones_a_contrario(id_iris1);

DROP TABLE nfa_a_contrario;
CREATE TABLE nfa_a_contrario (
	id_iris1 INTEGER NOT NULL,
	id_iris2 INTEGER NOT NULL,
	nfa FLOAT NOT NULL,
	intra_clase INTEGER NOT NULL,
	PRIMARY KEY(id_iris1, id_iris2)
);

