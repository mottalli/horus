#include "irisdatabasecuda.h"
#include "cudacommon.h"

IrisDatabaseCUDA::IrisDatabaseCUDA()
{
	this->dirty = true;
}

void IrisDatabaseCUDA::addTemplate(int templateId, const IrisTemplate& irisTemplate)
{
	IrisDatabase::addTemplate(templateId, irisTemplate);
	this->dirty = true;
}

void IrisDatabaseCUDA::deleteTemplate(int templateId)
{
	IrisDatabase::deleteTemplate(templateId);
	this->dirty = true;
}

void IrisDatabaseCUDA::doMatch(const IrisTemplate& irisTemplate, void (*statusCallback)(int), int nRots, int rotStep)
{
	// Unpack the templates
	vector<CvMat*> matTemplates(this->templates.size()), matMasks(this->templates.size());
	vector<uint8_t*> rawTemplates(this->templates.size()), rawMasks(this->templates.size());
	for (size_t i = 0; i < this->templates.size(); i++) {
		matTemplates[i] = this->templates[i]->getUnpackedTemplate();
		matMasks[i] = this->templates[i]->getUnpackedMask();

		assert(matTemplates[i]->step == matTemplates[i]->width);		// Indirect way of assuring it's CV_8U

		rawTemplates[i] = matTemplates[i]->data.ptr;
		rawMasks[i] = matMasks[i]->data.ptr;
	}

	loadDatabase(rawTemplates, rawMasks, matTemplates[0]->width, matTemplates[0]->height, &this->gpuDatabase);

	CvMat* template_ = irisTemplate.getUnpackedTemplate();
	CvMat* mask = irisTemplate.getUnpackedMask();

	doGPUMatch(template_->data.ptr, mask->data.ptr, &this->gpuDatabase, nRots, rotStep);

	// Free the memory
	for (size_t i = 0; i < this->templates.size(); i++) {
		cvReleaseMat(&matTemplates[i]);
		cvReleaseMat(&matMasks[i]);
	}
	cvReleaseMat(&template_);
	cvReleaseMat(&mask);

}
