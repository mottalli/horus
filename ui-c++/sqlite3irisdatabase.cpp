#include <sys/stat.h>
#include "sqlite3irisdatabase.h"
#include "processingthread.h"

using namespace horus::serialization;

extern ProcessingThread PROCESSING_THREAD;

SQLite3IrisDatabase::SQLite3IrisDatabase(const string& dbPath) :
	dbPath(dbPath)
{
	string sql;
	string dbFile = dbPath + "/base.db";

	this->db.open(dbFile);

	qDebug() << "Cargando base de datos...";

	SQlite3Database::Recordset rs = this->db.prepareStatement("SELECT id_iris,template FROM vw_base_iris WHERE entrada_valida=1").getRecordset();
	while (rs.next()) {
		int idTemplate = rs.at<int>(0);
		string serializedTemplate = rs.at<string>(1);

		if (serializedTemplate.length() == 0) {
			throw runtime_error("Se detectó una imagen no codificada");
		}

		this->addTemplate(idTemplate, serialization::unserializeIrisTemplate(serializedTemplate));
	}

	qDebug() << "Fin carga";

}

SQLite3IrisDatabase::~SQLite3IrisDatabase()
{
}

SQLite3IrisDatabase::IrisData SQLite3IrisDatabase::getIrisData(int irisId) const
{
	IrisData res;

	res.userId = -1;

	SQlite3Database::PreparedStatement stmt = this->db.prepareStatement("SELECT id_usuario,nombre,segmentacion,template,imagen FROM vw_base_iris WHERE id_iris=?");
	stmt << irisId;
	SQlite3Database::Recordset rs = stmt.getOne();

	if (rs.isAvailable()) {
		// Match
		int userId = rs.at<int>(0);
		string userName = rs.at<string>(1);
		string serializedSegmentation = rs.at<string>(2);
		string serializedTemplate = rs.at<string>(3);
		string imagePath = rs.at<string>(4);
		string fullPath = ( (imagePath[0] == '/') ? imagePath : this->dbPath + '/' + imagePath );

		res.userId = userId;
		res.userName = userName;
		res.segmentation = serialization::unserializeSegmentationResult(serializedSegmentation);
		res.irisTemplate = serialization::unserializeIrisTemplate(serializedTemplate);
		res.image = imread(fullPath, 1);
	} else {
		throw runtime_error("Invalid iris ID");
	}

	return res;
}

void SQLite3IrisDatabase::addUser(string userName, const IrisTemplate& irisTemplate, const SegmentationResult& segmentationResult, const Image& image)
{
	if (userName.empty()) {
		throw runtime_error("El nombre no puede estar vacio");
	}

	// Me fijo si hay otro con el mismo nombre
	SQlite3Database::PreparedStatement stmt = this->db.prepareStatement("SELECT * FROM usuarios WHERE LOWER(nombre) = LOWER(?)");
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

	string serializedImageTemplate = serialization::serializeIrisTemplate(imageTemplate);
	string serializedAverageTemplate = ( (averageTemplate) ? serialization::serializeIrisTemplate(*averageTemplate) : "");
	string serializedSegmentationResult = serialization::serializeSegmentationResult(segmentationResult);

	SQlite3Database::PreparedStatement stmt = this->db.prepareStatement("INSERT INTO base_iris(id_usuario,imagen,segmentacion,entrada_valida,image_template,average_template) \
						VALUES (?,?,?,?,?,?)");
	stmt << userId << filename << serializedSegmentationResult << 1 << serializedImageTemplate << serializedAverageTemplate;
	stmt.run();

	int irisId = this->db.lastInsertRowid();
	this->addTemplate(irisId, ((averageTemplate) ? *averageTemplate : imageTemplate));
}
