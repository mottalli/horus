#include <sys/stat.h>
#include "sqlite3irisdatabase.h"
#include "processingthread.h"

extern ProcessingThread PROCESSING_THREAD;

SQLite3IrisDatabase::SQLite3IrisDatabase(const string& dbPath) :
	dbPath(dbPath)
{
	string sql;
	string dbFile = dbPath + "/base.db";

	this->db.open(dbFile);

	qDebug() << "Cargando base de datos...";

	SQlite3Database::Recordset rs = this->db.prepareStatement("SELECT id_usuario, iris_template FROM usuarios NATURAL JOIN base_iris WHERE imagen_primaria=1").getRecordset();
	while (rs.next()) {
		int idUsuario = rs.at<int>(0);
		string serializedTemplate = rs.at<string>(1);

		if (serializedTemplate.length() == 0) {
			throw runtime_error("Se detectó una imagen no codificada");
		}

		this->addTemplate(idUsuario, Serializer::unserializeIrisTemplate(serializedTemplate));
	}

	qDebug() << "Fin carga";

}

SQLite3IrisDatabase::~SQLite3IrisDatabase()
{
}

SQLite3IrisDatabase::IrisData SQLite3IrisDatabase::getIrisData(int userId) const
{
	IrisData res;

	res.userId = -1;

	SQlite3Database::PreparedStatement stmt = this->db.prepareStatement("SELECT nombre,segmentacion,iris_template,imagen FROM usuarios NATURAL JOIN base_iris WHERE id_usuario=? AND imagen_primaria=1");
	stmt << userId;
	SQlite3Database::Recordset rs = stmt.getOne();

	if (rs.isAvailable()) {
		// Match
		string userName = rs.at<string>(0);
		string serializedSegmentation = rs.at<string>(1);
		string serializedTemplate = rs.at<string>(2);
		string imagePath = this->dbPath + '/' + rs.at<string>(3);

		res.userId = userId;
		res.userName = userName;
		res.segmentation = Serializer::unserializeSegmentationResult(serializedSegmentation);
		res.irisTemplate = Serializer::unserializeIrisTemplate(serializedTemplate);
		res.image = imread(imagePath, 1);
	}

	return res;
}

void SQLite3IrisDatabase::addUser(string userName, const IrisTemplate& irisTemplate, const SegmentationResult& segmentationResult, const Image& image)
{
	if (userName.empty()) {
		throw runtime_error("El nombre no puede estar vacío");
	}

	// Me fijo si hay otro con el mismo nombre
	SQlite3Database::PreparedStatement stmt = this->db.prepareStatement("SELECT id_usuario FROM usuarios WHERE LOWER(nombre) = LOWER(?)");
	stmt << userName;
	if (stmt.getOne().isAvailable()) {
		throw runtime_error("Ya existe un usuario en la base de datos con ese nombre");
	}

	try {
		// Inserto
		stmt = this->db.prepareStatement("INSERT INTO usuarios(nombre) VALUES(?)");
		stmt << userName;
		stmt.run();

		// Obtengo el ID insertado
		int userId = this->db.lastInsertRowid();

		// Guardo la imagen
		this->addImage(userId, image, segmentationResult, irisTemplate);
		this->addTemplate(userId, irisTemplate);
	} catch (SQLException ex) {
		throw ex;		//TODO - Manejar esto
	}
}

void SQLite3IrisDatabase::addImage(int userId, const Image& image, const SegmentationResult& segmentationResult, optional<IrisTemplate> averageTemplate)
{
	string filename, sql;

	for (int i = 1; ; i++) {
		filename = (boost::format("%i_%i.jpg") % userId % i).str();
		string fullFilename = (boost::format("%s/%s") % this->dbPath % filename).str();			// /path/to/db/<id>_<i>.jpg
		if (!filesystem::is_regular_file(fullFilename)) {			// Encontré un nombre disponible para el archivo
			imwrite(fullFilename, image);
			break;
		}
	}

	IrisTemplate imageTemplate = ::PROCESSING_THREAD.videoProcessor.irisEncoder.generateTemplate(image, segmentationResult);

	// Chequeo si es la primer imagen
	SQlite3Database::PreparedStatement stmt = this->db.prepareStatement("SELECT COUNT(*) FROM base_iris WHERE id_usuario=? AND imagen_primaria=1");
	stmt << userId;
	SQlite3Database::Recordset rs = stmt.getOne();
	int imagenPrimaria = (rs.at<int>(0) == 0) ? 1 : 0;

	string serializedImageTemplate = Serializer::serializeIrisTemplate(imageTemplate);
	string serializedAverageTemplate = ( (averageTemplate) ? Serializer::serializeIrisTemplate(*averageTemplate) : "");
	string serializedSegmentationResult = Serializer::serializeSegmentationResult(segmentationResult);

	stmt = this->db.prepareStatement("INSERT INTO base_iris(id_usuario,imagen_primaria,imagen,segmentacion,segmentacion_correcta,iris_template,average_template) \
						VALUES (?,?,?,?,?,?,?)");
	stmt << userId << imagenPrimaria << filename << serializedSegmentationResult << 1 << serializedImageTemplate << serializedAverageTemplate;
	stmt.run();
}
