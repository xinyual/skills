/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.agent.tools;

import static org.opensearch.agent.tools.utils.ToolHelper.getPPLTransportActionListener;
import static org.opensearch.ml.common.utils.StringUtils.gson;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.security.AccessController;
import java.security.PrivilegedExceptionAction;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.StringJoiner;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicBoolean;

import org.apache.commons.text.StringSubstitutor;
import org.json.JSONObject;
import org.opensearch.OpenSearchStatusException;
import org.opensearch.action.ActionRequest;
import org.opensearch.action.admin.indices.mapping.get.GetMappingsRequest;
import org.opensearch.action.bulk.BulkRequest;
import org.opensearch.action.search.SearchRequest;
import org.opensearch.action.update.UpdateRequest;
import org.opensearch.agent.indices.IndicesHelper;
import org.opensearch.agent.indices.SkillsIndexEnum;
import org.opensearch.agent.tools.utils.ToolHelper;
import org.opensearch.cluster.metadata.MappingMetadata;
import org.opensearch.common.xcontent.XContentHelper;
import org.opensearch.common.xcontent.json.JsonXContent;
import org.opensearch.core.action.ActionListener;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.core.xcontent.MediaTypeRegistry;
import org.opensearch.index.query.MatchAllQueryBuilder;
import org.opensearch.ml.common.FunctionName;
import org.opensearch.ml.common.dataset.remote.RemoteInferenceInputDataSet;
import org.opensearch.ml.common.input.MLInput;
import org.opensearch.ml.common.output.model.ModelTensor;
import org.opensearch.ml.common.output.model.ModelTensorOutput;
import org.opensearch.ml.common.output.model.ModelTensors;
import org.opensearch.ml.common.spi.tools.Tool;
import org.opensearch.ml.common.spi.tools.ToolAnnotation;
import org.opensearch.ml.common.transport.prediction.MLPredictionTaskAction;
import org.opensearch.ml.common.transport.prediction.MLPredictionTaskRequest;
import org.opensearch.search.SearchHit;
import org.opensearch.search.builder.SearchSourceBuilder;
import org.opensearch.sql.plugin.transport.PPLQueryAction;
import org.opensearch.sql.plugin.transport.TransportPPLQueryRequest;
import org.opensearch.sql.ppl.domain.PPLQueryRequest;
import org.opensearch.transport.client.Client;
import org.opensearch.transport.client.Requests;

import com.google.common.collect.ImmutableMap;
import com.google.common.hash.Hashing;

import lombok.Getter;
import lombok.Setter;
import lombok.extern.log4j.Log4j2;

@Log4j2
@Setter
@Getter
@ToolAnnotation(IndexSummarizeTool.TYPE)
public class IndexSummarizeTool implements Tool {
    public static final String TYPE = "IndexSummarizeTool";

    @Setter
    private Client client;

    private static final String DEFAULT_DESCRIPTION = "";

    @Setter
    @Getter
    private String name = TYPE;

    @Getter
    @Setter
    private String description = DEFAULT_DESCRIPTION;
    @Getter
    @Setter
    private String version;

    private IndicesHelper indicesHelper;

    private String modelId;

    private Map<String, Object> attributes;

    private String contextPrompt;

    private static Map<String, String> DEFAULT_PROMPT_DICT;

    static {
        DEFAULT_PROMPT_DICT = loadDefaultPromptDict();
    }

    public IndexSummarizeTool(Client client, IndicesHelper indicesHelper, String modelId, String contextPrompt) {
        this.client = client;
        this.indicesHelper = indicesHelper;
        this.modelId = modelId;
        if (!contextPrompt.isEmpty()) {
            this.contextPrompt = contextPrompt;
        } else {
            this.contextPrompt = DEFAULT_PROMPT_DICT.get("CLAUDE");
        }
    }

    @Override
    public String getType() {
        return TYPE;
    }

    @Override
    public boolean validate(Map<String, String> map) {
        return true;
    }

    @Override
    public <T> void run(Map<String, String> parameters, ActionListener<T> listener) {
        this.indicesHelper.initIndexIfAbsent(SkillsIndexEnum.SKILLS_INDEX_SUMMARY, ActionListener.wrap(indexCreated -> {
            if (!indexCreated) {
                listener.onFailure(new RuntimeException("No response to create ML Connector index"));
                return;
            }
            String indexName = parameters.get("index");

            ActionListener<String> promptCallingActionListener = ActionListener.wrap(prompt -> {
                log.info(prompt);
                RemoteInferenceInputDataSet inputDataSet = RemoteInferenceInputDataSet
                    .builder()
                    .parameters(Collections.singletonMap("prompt", prompt))
                    .build();
                ActionRequest request = new MLPredictionTaskRequest(
                    modelId,
                    MLInput.builder().algorithm(FunctionName.REMOTE).inputDataset(inputDataSet).build(),
                    null,
                    null
                );
                client.execute(MLPredictionTaskAction.INSTANCE, request, ActionListener.wrap(mlTaskResponse -> {
                    ModelTensorOutput modelTensorOutput = (ModelTensorOutput) mlTaskResponse.getOutput();
                    ModelTensors modelTensors = modelTensorOutput.getMlModelOutputs().get(0);
                    ModelTensor modelTensor = modelTensors.getMlModelTensors().get(0);
                    Map<String, String> dataAsMap = (Map<String, String>) modelTensor.getDataAsMap();
                    if (dataAsMap.get("response") == null) {
                        listener.onFailure(new IllegalStateException("Remote endpoint fails to inference."));
                        return;
                    }
                    String modelResponse = dataAsMap.get("response");
                    log.info("----------------");
                    log.info(modelResponse);
                    log.info("----------------");
                    String indexDescription = parseIndexDescription(modelResponse);
                    Map<String, Object> fieldDescription = parseFieldDescription(modelResponse);
                    BulkRequest bulkRequest = Requests.bulkRequest();
                    Map<String, Object> docMap = new HashMap<>();
                    docMap.put("index_name", indexName);
                    docMap.put("index_description", indexDescription);
                    docMap.put("field_description", fieldDescription);
                    bulkRequest
                        .add(
                            new UpdateRequest(SkillsIndexEnum.SKILLS_INDEX_SUMMARY.getIndexName(), generateDocId(indexName))
                                .doc(docMap, MediaTypeRegistry.JSON)
                                .docAsUpsert(true)
                        );
                    client.bulk(bulkRequest, ActionListener.wrap(r -> {
                        if (r.hasFailures()) {
                            log.error("Bulk create index summary embedding with failure {}", r.buildFailureMessage());
                        } else {
                            log.debug("Bulk create index summary embedding finished with {}", r.getTook());
                            listener
                                .onResponse(
                                    (T) AccessController
                                        .doPrivileged(
                                            (PrivilegedExceptionAction<String>) () -> gson
                                                .toJson(
                                                    Map.of("index_description", indexDescription, "field_description", fieldDescription)
                                                )
                                        )
                                );
                        }
                    }, exception -> log.error("Bulk create index summary embedding failed", exception)));

                }, e -> {
                    log.error("fail to call LLM");
                    listener.onFailure(new IllegalStateException(e.getMessage()));
                }));
            }, e -> {
                log.error("fail to get unique field");
                listener.onFailure(new IllegalStateException(e.getMessage()));
            }

            );

            GetMappingsRequest getMappingsRequest = new GetMappingsRequest().indices(indexName);
            client.admin().indices().getMappings(getMappingsRequest, ActionListener.wrap(getMappingsResponse -> {
                Map<String, MappingMetadata> mappings = getMappingsResponse.getMappings();
                if (mappings.isEmpty()) {
                    throw new IllegalArgumentException("No matching mapping with index name: " + indexName);
                }
                String firstIndexName = (String) mappings.keySet().toArray()[0];
                SearchSourceBuilder searchSourceBuilder = new SearchSourceBuilder();
                searchSourceBuilder.size(1).query(new MatchAllQueryBuilder());
                SearchRequest searchRequest = new SearchRequest(new String[] { indexName }, searchSourceBuilder);
                client.search(searchRequest, ActionListener.wrap(searchResponse -> {
                    SearchHit[] searchHits = searchResponse.getHits().getHits();
                    parallelGetUniqueFields(indexName, mappings.get(firstIndexName), searchHits, promptCallingActionListener);
                }, e -> {
                    log.error("fail to call dsl");
                    listener.onFailure(new IllegalStateException(e.getMessage()));
                }));
            }, e -> {
                log.error("fail to get mapping");
                listener.onFailure(new IllegalStateException(e.getMessage()));
            }

            ));

        }, e -> {
            log.error("fail to create index");

        }

        ));

    }

    private String parseIndexDescription(String modelResponse) {
        String[] tmpList = modelResponse.split("<total_summarization>");

        return tmpList[tmpList.length - 1].split("</total_summarization>")[0];
    }

    private Map<String, Object> parseFieldDescription(String modelResponse) {
        String[] tmpList = modelResponse.split("<column_summarization>");
        String content = tmpList[tmpList.length - 1].split("</column_summarization>")[0];
        log.info("field desc content");
        log.info(content);
        Map<String, Object> field2Desc = new HashMap<>();
        String[] lines = content.split("\n");

        // Process each line
        for (String line : lines) {
            line = line.trim(); // Remove leading/trailing whitespace

            // Split the line into name and description using the first ":"
            String[] parts = line.split(":", 2); // Use "2" to limit splits to two parts
            if (parts.length == 2) {
                String name = parts[0];
                String desc = parts[1];
                field2Desc.put(name, desc);
            }
        }
        return field2Desc;
    }

    private void parallelGetUniqueFields(
        String indexName,
        MappingMetadata mappingMetadata,
        SearchHit[] sampleResult,
        ActionListener<String> actionListener
    ) {
        Map<String, Object> mappingSource = (Map<String, Object>) mappingMetadata.getSourceAsMap().get("properties");
        Map<String, String> fieldsToType = new HashMap<>();
        ToolHelper.extractFieldNamesTypes(mappingSource, fieldsToType, "", false);
        CountDownLatch countDownLatch = new CountDownLatch(fieldsToType.size());
        ConcurrentHashMap<String, Object> resultsMap = new ConcurrentHashMap<>();
        AtomicBoolean noErrors = new AtomicBoolean(true);
        ActionListener<Map<String, Object>> countDownActionListener = ActionListener.wrap(b -> {
            countDownLatch.countDown();
            resultsMap.putAll(b);
            if (countDownLatch.getCount() == 0) {
                if (noErrors.get()) {
                    String prompt = formatPrompt(indexName, mappingMetadata, sampleResult, resultsMap);
                    log.info("-------------");
                    log.info(prompt);
                    log.info("-------------");
                    actionListener.onResponse(prompt);
                } else {
                    actionListener.onFailure(new OpenSearchStatusException("something wrong", RestStatus.CONFLICT));
                }
            }
        }, e -> {
            countDownLatch.countDown();
            noErrors.set(false);
            actionListener.onFailure(new OpenSearchStatusException(e.getMessage(), RestStatus.CONFLICT));
        });
        for (Map.Entry<String, String> entry : fieldsToType.entrySet()) {
            String fieldName = entry.getKey();
            getUniqueForSingleField(indexName, fieldName, countDownActionListener);
        }

    }

    private void getUniqueForSingleField(String indexName, String fieldName, ActionListener<Map<String, Object>> actionListener) {
        String ppl = "source=" + indexName + "| dedup `" + fieldName + "` | fields `" + fieldName + "` | head 5";
        JSONObject jsonContent = new JSONObject(ImmutableMap.of("query", ppl));
        PPLQueryRequest pplQueryRequest = new PPLQueryRequest(ppl, jsonContent, null, "jdbc");
        TransportPPLQueryRequest transportPPLQueryRequest = new TransportPPLQueryRequest(pplQueryRequest);
        client
            .execute(
                PPLQueryAction.INSTANCE,
                transportPPLQueryRequest,
                getPPLTransportActionListener(ActionListener.wrap(transportPPLQueryResponse -> {
                    String results = transportPPLQueryResponse.getResult();
                    Map<String, Object> pplResultMap = XContentHelper.convertToMap(JsonXContent.jsonXContent, results, true);
                    Map<String, Object> resultMap = ImmutableMap.of(fieldName, pplResultMap.getOrDefault("datarows", null));
                    actionListener.onResponse(resultMap);
                }, e -> {
                    String pplError = "execute ppl:" + ppl + ", get error: " + e.getMessage();
                    Exception exception = new Exception(pplError);
                    actionListener.onFailure(exception);
                }))
            );
    }

    private String formatPrompt(
        String indexName,
        MappingMetadata mappingMetadata,
        SearchHit[] samples,
        Map<String, Object> fieldUniqueValue
    ) {
        String tableInfo = constructMappingInfo(mappingMetadata, fieldUniqueValue);
        Map<String, String> indexInfo = ImmutableMap
            .of("tableInfo", tableInfo, "indexName", indexName, "samples", Arrays.toString(samples));
        StringSubstitutor substitutor = new StringSubstitutor(indexInfo, "${indexInfo.", "}");
        return substitutor.replace(contextPrompt);
    }

    private String constructMappingInfo(MappingMetadata mappingMetadata, Map<String, Object> fieldUniqueValue) {
        Map<String, Object> mappingSource = (Map<String, Object>) mappingMetadata.getSourceAsMap().get("properties");
        if (Objects.isNull(mappingSource)) {
            throw new IllegalArgumentException(
                "The querying index doesn't have mapping metadata, please add data to it or using another index."
            );
        }
        Map<String, String> fieldsToType = new HashMap<>();
        ToolHelper.extractFieldNamesTypes(mappingSource, fieldsToType, "", false);
        StringJoiner tableInfoJoiner = new StringJoiner("\n");
        List<String> sortedKeys = new ArrayList<>(fieldsToType.keySet());
        Collections.sort(sortedKeys);
        for (String key : sortedKeys) {
            String line = "- " + key + ": " + fieldsToType.get(key) + " (" + fieldUniqueValue.get(key) + ")";
            tableInfoJoiner.add(line);
        }
        return tableInfoJoiner.toString();
    }

    public static class Factory implements Tool.Factory<IndexSummarizeTool> {
        private Client client;

        private IndicesHelper indicesHelper;

        private static IndexSummarizeTool.Factory INSTANCE;

        public static IndexSummarizeTool.Factory getInstance() {
            if (INSTANCE != null) {
                return INSTANCE;
            }
            synchronized (IndexSummarizeTool.class) {
                if (INSTANCE != null) {
                    return INSTANCE;
                }
                INSTANCE = new IndexSummarizeTool.Factory();
                return INSTANCE;
            }
        }

        public void init(Client client, IndicesHelper indicesHelper) {
            this.client = client;
            this.indicesHelper = indicesHelper;
        }

        @Override
        public IndexSummarizeTool create(Map<String, Object> map) {
            return new IndexSummarizeTool(client, indicesHelper, (String) map.get("model_id"), (String) map.getOrDefault("prompt", ""));
        }

        @Override
        public String getDefaultDescription() {
            return DEFAULT_DESCRIPTION;
        }

        @Override
        public String getDefaultType() {
            return TYPE;
        }

        @Override
        public String getDefaultVersion() {
            return null;
        }

    }

    @SuppressWarnings("unchecked")
    private static Map<String, String> loadDefaultPromptDict() {
        try (InputStream searchResponseIns = IndexSummarizeTool.class.getResourceAsStream("IndexSummarizeDefaultPrompt.json")) {
            if (searchResponseIns != null) {
                String defaultPromptContent = new String(searchResponseIns.readAllBytes(), StandardCharsets.UTF_8);
                return gson.fromJson(defaultPromptContent, Map.class);
            }
        } catch (IOException e) {
            log.error("Failed to load default prompt dict", e);
        }
        return new HashMap<>();
    }

    private String generateDocId(String indexName) {
        return Hashing.sha256().hashString(indexName, StandardCharsets.UTF_8).toString();
    }

}
