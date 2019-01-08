//
// traffic_sign_with_class_weights.swift
//
// This file was automatically generated and should not be edited.
//

import CoreML


/// Model Prediction Input Type
@available(macOS 10.13, iOS 11.0, tvOS 11.0, watchOS 4.0, *)
class traffic_sign_with_class_weightsInput : MLFeatureProvider {

    /// input_1 as color (kCVPixelFormatType_32BGRA) image buffer, 112 pixels wide by 112 pixels high
    var input_1: CVPixelBuffer

    var featureNames: Set<String> {
        get {
            return ["input_1"]
        }
    }
    
    func featureValue(for featureName: String) -> MLFeatureValue? {
        if (featureName == "input_1") {
            return MLFeatureValue(pixelBuffer: input_1)
        }
        return nil
    }
    
    init(input_1: CVPixelBuffer) {
        self.input_1 = input_1
    }
}

/// Model Prediction Output Type
@available(macOS 10.13, iOS 11.0, tvOS 11.0, watchOS 4.0, *)
class traffic_sign_with_class_weightsOutput : MLFeatureProvider {

    /// Source provided by CoreML

    private let provider : MLFeatureProvider


    /// activation_1 as dictionary of strings to doubles
    lazy var activation_1: [String : Double] = {
        [unowned self] in return self.provider.featureValue(for: "activation_1")!.dictionaryValue as! [String : Double]
    }()

    /// classLabel as string value
    lazy var classLabel: String = {
        [unowned self] in return self.provider.featureValue(for: "classLabel")!.stringValue
    }()

    var featureNames: Set<String> {
        return self.provider.featureNames
    }
    
    func featureValue(for featureName: String) -> MLFeatureValue? {
        return self.provider.featureValue(for: featureName)
    }

    init(activation_1: [String : Double], classLabel: String) {
        self.provider = try! MLDictionaryFeatureProvider(dictionary: ["activation_1" : MLFeatureValue(dictionary: activation_1 as [AnyHashable : NSNumber]), "classLabel" : MLFeatureValue(string: classLabel)])
    }

    init(features: MLFeatureProvider) {
        self.provider = features
    }
}


/// Class for model loading and prediction
@available(macOS 10.13, iOS 11.0, tvOS 11.0, watchOS 4.0, *)
class traffic_sign_with_class_weights {
    var model: MLModel

/// URL of model assuming it was installed in the same bundle as this class
    class var urlOfModelInThisBundle : URL {
        let bundle = Bundle(for: traffic_sign_with_class_weights.self)
        return bundle.url(forResource: "traffic_sign_with_class_weights", withExtension:"mlmodelc")!
    }

    /**
        Construct a model with explicit path to mlmodelc file
        - parameters:
           - url: the file url of the model
           - throws: an NSError object that describes the problem
    */
    init(contentsOf url: URL) throws {
        self.model = try MLModel(contentsOf: url)
    }

    /// Construct a model that automatically loads the model from the app's bundle
    convenience init() {
        try! self.init(contentsOf: type(of:self).urlOfModelInThisBundle)
    }

    /**
        Construct a model with configuration
        - parameters:
           - configuration: the desired model configuration
           - throws: an NSError object that describes the problem
    */
    @available(macOS 10.14, iOS 12.0, tvOS 12.0, watchOS 5.0, *)
    convenience init(configuration: MLModelConfiguration) throws {
        try self.init(contentsOf: type(of:self).urlOfModelInThisBundle, configuration: configuration)
    }

    /**
        Construct a model with explicit path to mlmodelc file and configuration
        - parameters:
           - url: the file url of the model
           - configuration: the desired model configuration
           - throws: an NSError object that describes the problem
    */
    @available(macOS 10.14, iOS 12.0, tvOS 12.0, watchOS 5.0, *)
    init(contentsOf url: URL, configuration: MLModelConfiguration) throws {
        self.model = try MLModel(contentsOf: url, configuration: configuration)
    }

    /**
        Make a prediction using the structured interface
        - parameters:
           - input: the input to the prediction as traffic_sign_with_class_weightsInput
        - throws: an NSError object that describes the problem
        - returns: the result of the prediction as traffic_sign_with_class_weightsOutput
    */
    func prediction(input: traffic_sign_with_class_weightsInput) throws -> traffic_sign_with_class_weightsOutput {
        return try self.prediction(input: input, options: MLPredictionOptions())
    }

    /**
        Make a prediction using the structured interface
        - parameters:
           - input: the input to the prediction as traffic_sign_with_class_weightsInput
           - options: prediction options 
        - throws: an NSError object that describes the problem
        - returns: the result of the prediction as traffic_sign_with_class_weightsOutput
    */
    func prediction(input: traffic_sign_with_class_weightsInput, options: MLPredictionOptions) throws -> traffic_sign_with_class_weightsOutput {
        let outFeatures = try model.prediction(from: input, options:options)
        return traffic_sign_with_class_weightsOutput(features: outFeatures)
    }

    /**
        Make a prediction using the convenience interface
        - parameters:
            - input_1 as color (kCVPixelFormatType_32BGRA) image buffer, 112 pixels wide by 112 pixels high
        - throws: an NSError object that describes the problem
        - returns: the result of the prediction as traffic_sign_with_class_weightsOutput
    */
    func prediction(input_1: CVPixelBuffer) throws -> traffic_sign_with_class_weightsOutput {
        let input_ = traffic_sign_with_class_weightsInput(input_1: input_1)
        return try self.prediction(input: input_)
    }

    /**
        Make a batch prediction using the structured interface
        - parameters:
           - inputs: the inputs to the prediction as [traffic_sign_with_class_weightsInput]
           - options: prediction options 
        - throws: an NSError object that describes the problem
        - returns: the result of the prediction as [traffic_sign_with_class_weightsOutput]
    */
    @available(macOS 10.14, iOS 12.0, tvOS 12.0, watchOS 5.0, *)
    func predictions(inputs: [traffic_sign_with_class_weightsInput], options: MLPredictionOptions = MLPredictionOptions()) throws -> [traffic_sign_with_class_weightsOutput] {
        let batchIn = MLArrayBatchProvider(array: inputs)
        let batchOut = try model.predictions(from: batchIn, options: options)
        var results : [traffic_sign_with_class_weightsOutput] = []
        results.reserveCapacity(inputs.count)
        for i in 0..<batchOut.count {
            let outProvider = batchOut.features(at: i)
            let result =  traffic_sign_with_class_weightsOutput(features: outProvider)
            results.append(result)
        }
        return results
    }
}
