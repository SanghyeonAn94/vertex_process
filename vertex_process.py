import torch
import torch.nn as nn

class VertexProcessor(nn.Module):
    def __init__(self):
        super(VertexProcessor, self).__init__()

    def forward(self, vertices_a: torch.Tensor, vertices_b: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:  
        # Input shapes:  
        # vertices_a: (batch_size, 10475, 3)  
        # vertices_b: (batch_size, 10475, 3)  
        # scale: (batch_size, 1)  
        batch_size = vertices_a.shape[0]
        # 1. Sum the vertices  
        summed_vertices = vertices_a + vertices_b  
        
        # 2. Normalize the vertices  
        # Calculate mean and std for each batch  
        mean = torch.mean(summed_vertices, dim=1, keepdim=True)  
        std = torch.std(summed_vertices, dim=1, keepdim=True)  
        normalized_vertices = (summed_vertices - mean) / (std + 1e-8)  
        
        # 3. Find minimum points and shift to origin  
        min_points = torch.min(normalized_vertices, dim=1)[0]  # (batch_size, 3)  
        shifted_vertices = normalized_vertices - min_points.unsqueeze(1)  
        
        # 4. Find maximum points  
        max_points = torch.max(shifted_vertices, dim=1)[0]  # (batch_size, 3)  
        
        # 5. Sum the maximum points and divide by scale
        # Assuming scale is a tensor of shape (batch_size, 1)
        # Result shape: (batch_size, 1)  
        result = torch.sum(max_points, dim=1) / scale.squeeze(1)  
        
        return result
        
def create_and_test_model():
    # 모델 인스턴스 생성
    model = VertexProcessor()
    model.eval()

    # 테스트 데이터 생성  
    batch_size = 2  
    vertices_a = torch.randn(batch_size, 10475, 3)  
    vertices_b = torch.randn(batch_size, 10475, 3)  
    scale = torch.tensor([2.0, 3.0]).reshape(batch_size, 1) 
    
    # ONNX 내보내기  
    torch.onnx.export(  
        model,  
        (vertices_a, vertices_b, scale),  
        r"./vertex_processor.onnx",
        input_names=['vertices_a', 'vertices_b', 'scale'],  
        output_names=['output'],  
        dynamic_axes={  
            'vertices_a': {0: 'batch_size'},  
            'vertices_b': {0: 'batch_size'},  
            'scale': {0: 'batch_size'},  
            'output': {0: 'batch_size'}  
        },  
        opset_version=17  
    )  
    
    # 테스트 실행  
    with torch.no_grad():  
        output = model(vertices_a, vertices_b, scale)  
        print(f"Output shape: {output.shape}")  
        print(f"Output values: {output}")


if __name__ == "__main__":
    create_and_test_model()