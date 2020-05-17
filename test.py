from radiomics import featureextractor

extractor = featureextractor.RadiomicsFeatureExtractor()
extractor.disableAllFeatures()
extractor.enableFeatureClassByName('glcm')


print "Enabled features:\n\t", extractor.enabledFeatures